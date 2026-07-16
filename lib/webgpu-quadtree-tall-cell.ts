import { adaptivePressureCellTopology, buildMlsProjectionRows, buildQuadtree, buildVariationalSystem, populateTallPressureGrid, populateTallPressureGridFromLeafProfiles, quadtreeFromPackedCells, quadtreeSizingFromVelocityAndSurface, signedDistanceFromVolume, type QuadtreeGrid, type TallPressureGrid, type TallPressureSample, type VariationalBody, type VariationalSystem } from "./quadtree-tall-cell-grid";
import { damBreakFractions } from "./initial-fluid";
import { insidePrimitive } from "./fluid-rigid-coupling";
import { boundingRadius, quaternionRotate, type RigidBodyState } from "./rigid-body";
import type { SceneDescription, Vec3 } from "./model";
import { WebGPUQuadtreeBuilder, WebGPUQuadtreeSurfaceState, type WebGPUQuadtreeConstructionCache, type WebGPUQuadtreeSurfaceCache } from "./webgpu-quadtree-builder";

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
  surface?: WebGPUQuadtreeSurfaceCache;
  cpuWorker?: { postMessage(message: unknown): void; terminate(): unknown };
  cpuWorkerSequence?: number;
  cpuWorkerPending?: Map<number, { resolve: (value: PreparedProjectionCPU) => void; reject: (reason: unknown) => void }>;
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
  /** IC(0) is paper-conformant; the other choices preserve the operator and tolerance stop. */
  preconditioner?: "ic0" | "jacobi" | "line" | "poly";
  /** Degree of the damped-Jacobi Neumann polynomial (2--4). */
  polynomialDegree?: number;
  /** Internal feedback carried across topology rebuilds. */
  iterationBudgetHint?: number;
  /** Internal exponential moving average of iterations-to-tolerance. */
  iterationEmaHint?: number;
  /** Opt-in timestamp-query breakdown of setup / early iterations / remainder / projection. */
  debugPressureTimings?: boolean;
  debugPressureFirstIterations?: number;
}

export interface QuadtreeIterationBudget {
  hardBudget: number;
  encodedBudget: number;
  ema: number;
}

/** Pure helper kept separate from GPU state so budget feedback is unit-testable. */
export function quadtreeIterationBudget(dofCount: number, options: Pick<QuadtreeTallCellProjectionOptions, "pressureIterations" | "iterationBudgetHint" | "iterationEmaHint">): QuadtreeIterationBudget {
  const hardBudget = Math.max(options.pressureIterations, Math.min(2048, Math.ceil(4 * Math.sqrt(Math.max(1, dofCount)))));
  const ema = Math.max(1, options.iterationEmaHint ?? options.iterationBudgetHint ?? hardBudget / 1.5);
  const hinted = options.iterationBudgetHint ?? Math.ceil(1.5 * ema);
  return { hardBudget, encodedBudget: Math.max(Math.min(32, hardBudget), Math.min(hardBudget, Math.ceil(hinted))), ema };
}

export function nextQuadtreeIterationBudget(current: QuadtreeIterationBudget, used: number, converged: boolean): QuadtreeIterationBudget {
  const boundedUsed = Math.max(0, Math.min(current.hardBudget, Math.round(used)));
  const ema = current.ema * 0.75 + boundedUsed * 0.25;
  const target = !converged && boundedUsed >= current.encodedBudget
    ? Math.max(current.encodedBudget + 1, current.encodedBudget * 2)
    : 1.5 * ema;
  return {
    hardBudget: current.hardBudget,
    encodedBudget: Math.max(Math.min(32, current.hardBudget), Math.min(current.hardBudget, Math.ceil(target))),
    ema
  };
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
  cpuICFactorization_ms?: number;
  cpuResourceUpload_ms?: number;
  topologyReused?: boolean;
  topologyReuseCount?: number;
  pressureIterationsUsed?: number;
  pressureIterationBudget?: number;
  pressureIterationHardBudget?: number;
  pressureConverged?: boolean;
  factorLevelCount?: number;
  pressurePhaseTimings?: { setup_ms: number; firstIterations_ms: number; remainingIterations_ms: number; project_ms: number };
  /** Bytes read back for an update (leaf-centre phi profiles + compact 2D leaves + diagnostics). */
  topologyReadbackBytes?: number;
}

interface ProjectionResources {
  velocityIn: GPUTexture;
  velocityOut: GPUTexture;
  volume: GPUTexture;
  levelSet?: WebGPUQuadtreeSurfaceState;
}

interface ProjectionFields {
  phi?: Float32Array;
  velocity?: Vec3[];
  quadtree?: QuadtreeGrid;
  pressureGrid?: TallPressureGrid;
  topologyWords?: Uint32Array;
  prepared?: PreparedProjectionCPU;
}

export interface QuadtreeCPUPreparationInput {
  scene: SceneDescription;
  dims: { nx: number; ny: number; nz: number };
  options: QuadtreeTallCellProjectionOptions;
  packedCells: Uint32Array;
  columnProfiles: Float32Array;
  coupling?: QuadtreeRigidCoupling;
}

export interface PreparedProjectionCPU {
  quadtree: QuadtreeGrid;
  pressureGrid: TallPressureGrid;
  topologyWords: Uint32Array;
  packed: ReturnType<typeof WebGPUQuadtreeTallCellProjection.packSystem>;
  displacedVolumes: number[];
  dofCount: number;
  faceCount: number;
  ghostFaceCount: number;
  maximumFluidScale: number;
  tallSegmentCount: number;
  quadtreeDecode_ms: number;
  tallGrid_ms: number;
  variationalAssembly_ms: number;
  systemPack_ms: number;
  icFactorization_ms: number;
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

function displacedVolumesForGrid(grid: TallPressureGrid, phi: Float32Array | undefined, solidFields: ReturnType<typeof solidFieldsFromBodies>, bodyCount: number, nx: number, ny: number, h: Vec3) {
  const volumes = Array.from({ length: bodyCount }, () => 0);
  if (!solidFields) return volumes;
  const liquidByLeafY = phi ? undefined : new Uint8Array(grid.quadtree.leaves.length * ny);
  if (liquidByLeafY) for (const segment of grid.segments) {
    const liquid = grid.samples[segment.bottomSample].liquid ? 1 : 0;
    for (let y = segment.firstY; y <= segment.lastY; y += 1) liquidByLeafY[segment.leaf * ny + y] = liquid;
  }
  const cellVolume = h.x * h.y * h.z;
  for (let index = 0; index < solidFields.solidFraction.length; index += 1) {
    const owner = solidFields.solidOwner[index];
    if (owner < 0) continue;
    const x = index % nx, y = Math.floor(index / nx) % ny, z = Math.floor(index / (nx * ny));
    const liquid = phi ? phi[index] < 0 : liquidByLeafY![grid.quadtree.leafAt[x + nx * z] * ny + y] !== 0;
    if (liquid) volumes[owner] += solidFields.solidFraction[index] * cellVolume;
  }
  return volumes;
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
struct Params { dims: vec4u, cell: vec4f, counts: vec4u, solve: vec4f, coupling: vec4u, couplingCounts: vec4u, precondition: vec4u }
struct Face { nodes: vec4u, coefficients: vec4f, bounds: vec4u, packed: u32, solidFlux: f32, weights: vec2f, sampleCells: vec4u, sampleSpans: vec4u, flux: f32 }
struct Entry { face: u32, coefficient: f32 }
alias SolverField = u32;
const PRESSURE: SolverField = 0u; const BEST_PRESSURE: SolverField = 1u;
const RESIDUAL: SolverField = 2u; const DIRECTION: SolverField = 3u;
const PRECONDITIONED: SolverField = 4u; const MATRIX_DIRECTION: SolverField = 5u;
const DIAGONAL: SolverField = 6u; const ACTIVE_FLAG: SolverField = 7u;
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var volumeIn: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> faces: array<Face>;
@group(0) @binding(4) var<storage, read> rowOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> rowEntries: array<Entry>;
@group(0) @binding(6) var<storage, read_write> matrixWords: array<u32>;
@group(0) @binding(7) var cellProjection: texture_3d<f32>;
@group(0) @binding(8) var<storage, read_write> state: array<u32>;
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
fn stateIndex(row: u32, field: SolverField) -> u32 { return field * dofs() + row; }
fn stateF(row: u32, field: SolverField) -> f32 { return bitcast<f32>(state[stateIndex(row, field)]); }
fn setStateF(row: u32, field: SolverField, value: f32) { state[stateIndex(row, field)] = bitcast<u32>(value); }
fn addStateF(row: u32, field: SolverField, value: f32) { setStateF(row, field, stateF(row, field) + value); }
fn matrixEntryBase(entry: u32) -> u32 { return dofs() + 1u + 4u * entry; }
fn matrixStart(row: u32) -> u32 { return matrixWords[row]; }
fn matrixNode(entry: u32) -> u32 { return matrixWords[matrixEntryBase(entry)]; }
fn matrixFaceSlot(entry: u32) -> u32 { return matrixWords[matrixEntryBase(entry) + 1u]; }
fn matrixCoefficient(entry: u32) -> f32 { return bitcast<f32>(matrixWords[matrixEntryBase(entry) + 2u]); }
fn matrixBaseCoefficient(entry: u32) -> f32 { return bitcast<f32>(matrixWords[matrixEntryBase(entry) + 3u]); }
fn setMatrixCoefficient(entry: u32, value: f32) { matrixWords[matrixEntryBase(entry) + 2u] = bitcast<u32>(value); }
fn auxWord(index: u32) -> u32 {
  let texel = index / 4u; let packed = textureLoad(factorAux, vec2i(i32(texel % params.dims.w), i32(texel / params.dims.w)), 0);
  return packed[index % 4u];
}
fn auxEntry(wordOffset: u32, index: u32) -> Entry { return Entry(auxWord(wordOffset + 2u * index), bitcast<f32>(auxWord(wordOffset + 2u * index + 1u))); }
fn faceSpan(face: Face) -> u32 { return face.packed & 0xffffu; }
fn faceAxis(face: Face) -> u32 { return (face.packed >> 16u) & 0x3u; }
fn faceNodeCount(face: Face) -> u32 { return (face.packed >> 18u) & 0x7u; }
fn faceGhost(face: Face) -> bool { return ((face.packed >> 21u) & 0x1u) != 0u; }
fn faceSlotLiquid(face: Face, slot: u32) -> bool { return ((face.packed >> (22u + slot)) & 0x1u) != 0u; }
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
  var face = faces[faceId]; var all = 0.0; var liquid = 0.0; var allLiquid = true; var liquidMask = 0u;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let phi = faceSamplePhi(face, slot); let term = face.coefficients[slot] * phi; all += term;
    if (phi < 0.0) { liquid += term; liquidMask |= 1u << slot; } else { allLiquid = false; }
  }
  var scale = 1.0;
  if (!allLiquid) { scale = select(min(100.0, max(0.0, all / liquid)), 0.0, abs(liquid) < 1e-12); }
  face.weights.y = face.weights.x * scale;
  face.packed = (face.packed & 0x003fffffu) | (liquidMask << 22u);
  face.flux = face.weights.x * faceVelocity(face) + face.solidFlux;
  faces[faceId] = face;
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
    let node = face.nodes[slot]; if (node != 0xffffffffu && faceSlotLiquid(face, slot)) { result += face.coefficients[slot] * stateF(node, DIRECTION); }
  }
  return result;
}
@compute @workgroup_size(128)
fn refreshRows(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; } var rowIsActive = false;
  for (var entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; entry += 1u) {
    let face = faces[rowEntries[entry].face];
    for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
      if (face.nodes[slot] == row && faceSlotLiquid(face, slot)) { rowIsActive = true; }
    }
  }
  state[stateIndex(row, ACTIVE_FLAG)] = select(0u, 1u, rowIsActive);
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) {
    let packed = matrixFaceSlot(entry); let face = faces[packed & 0x3fffffffu]; let slot = packed >> 30u;
    let coefficient = select(0.0, matrixBaseCoefficient(entry) * face.weights.y, rowIsActive && faceSlotLiquid(face, slot));
    setMatrixCoefficient(entry, coefficient);
  }
}
fn dofActive(row: u32) -> bool { return state[stateIndex(row, ACTIVE_FLAG)] != 0u; }
fn rowProduct(row: u32) -> f32 {
  if (!dofActive(row)) { return stateF(row, DIRECTION); }
  var sum = 0.0;
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { sum += matrixCoefficient(entry) * stateF(matrixNode(entry), DIRECTION); }
  return sum;
}
fn initializeRow(row: u32) {
  if (!dofActive(row)) {
    setStateF(row, PRESSURE, 0.0); setStateF(row, BEST_PRESSURE, 0.0); setStateF(row, RESIDUAL, 0.0); setStateF(row, DIAGONAL, 1.0);
    setStateF(row, PRECONDITIONED, 0.0); setStateF(row, DIRECTION, 0.0); setStateF(row, MATRIX_DIRECTION, 0.0); return;
  }
  var rhs = 0.0; var diag = 0.0;
  for (var entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; entry += 1u) {
    let item = rowEntries[entry]; let face = faces[item.face];
    // The face flux is A u_fluid (from the staged texture) plus the
    // CPU-integrated (1-A) u_solid constraint flux of moving rigid bodies.
    rhs += item.coefficient * face.flux;
  }
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { if (matrixNode(entry) == row) { diag += matrixCoefficient(entry); } }
  setStateF(row, PRESSURE, 0.0); setStateF(row, BEST_PRESSURE, 0.0); setStateF(row, RESIDUAL, rhs); setStateF(row, DIAGONAL, max(diag, 1e-12));
  setStateF(row, PRECONDITIONED, 0.0); setStateF(row, DIRECTION, 0.0); setStateF(row, MATRIX_DIRECTION, 0.0);
}
@compute @workgroup_size(128)
fn initialize(@builtin(global_invocation_id) gid: vec3u) { if (gid.x < dofs()) { initializeRow(gid.x); } }
fn applyPrecondition(lid: u32, solveActive: bool) {
  if (solveActive) { for (var row = lid; row < dofs(); row += 256u) { setStateF(row, PRECONDITIONED, stateF(row, RESIDUAL)); } }
  storageBarrier(); workgroupBarrier();
  // The IC graph is level-scheduled so every row within a level is
  // independent. One workgroup supplies the required global ordering while
  // evaluating up to 256 triangular rows concurrently.
  for (var level = 0u; level < params.counts.w; level += 1u) {
    let levelsOffset = bitcast<u32>(params.solve.y); let rowOffsetsOffset = bitcast<u32>(params.solve.z); let rowEntriesOffset = bitcast<u32>(params.solve.w);
    let range = vec2u(auxWord(levelsOffset + 4u * level), auxWord(levelsOffset + 4u * level + 1u));
    if (solveActive) { for (var slot = range.x + lid; slot < range.y; slot += 256u) {
      let row = auxWord(slot); var value = stateF(row, PRECONDITIONED);
      for (var entry = auxWord(rowOffsetsOffset + row); entry < auxWord(rowOffsetsOffset + row + 1u); entry += 1u) {
        let factor = auxEntry(rowEntriesOffset, entry); value -= factor.coefficient * stateF(factor.face, PRECONDITIONED);
      }
      setStateF(row, PRECONDITIONED, value * bitcast<f32>(factorColumns[row].y));
    } }
    storageBarrier(); workgroupBarrier();
  }
  for (var level = 0u; level < params.counts.w; level += 1u) {
    let levelsOffset = bitcast<u32>(params.solve.y);
    let range = vec2u(auxWord(levelsOffset + 4u * level + 2u), auxWord(levelsOffset + 4u * level + 3u));
    if (solveActive) { for (var slot = range.x + lid; slot < range.y; slot += 256u) {
      let column = auxWord(slot); var value = stateF(column, PRECONDITIONED);
      for (var entry = factorColumns[column].x; entry < factorColumns[column + 1u].x; entry += 1u) {
        let factor = factorEntries[entry]; value -= factor.coefficient * stateF(factor.face, PRECONDITIONED);
      }
      setStateF(column, PRECONDITIONED, value * bitcast<f32>(factorColumns[column].y));
    } }
    storageBarrier(); workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn precondition(@builtin(local_invocation_id) lid: vec3u) {
  applyPrecondition(lid.x, !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3]));
}
@compute @workgroup_size(128)
fn preconditionJacobi(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row < dofs() && !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3])) {
    setStateF(row, PRECONDITIONED, stateF(row, RESIDUAL) / stateF(row, DIAGONAL));
  }
}
fn matrixValue(row: u32, node: u32) -> f32 {
  var value = 0.0;
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { if (matrixNode(entry) == node) { value += matrixCoefficient(entry); } }
  return value;
}
@compute @workgroup_size(128)
fn preconditionLine(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.precondition.z || (scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3])) { return; }
  let first = auxWord(params.precondition.x + gid.x); let end = auxWord(params.precondition.x + gid.x + 1u);
  var previous = 0xffffffffu; var previousC = 0.0; var previousD = 0.0;
  for (var slot = first; slot < end; slot += 1u) {
    let row = auxWord(params.precondition.y + slot); var next = 0xffffffffu; if (slot + 1u < end) { next = auxWord(params.precondition.y + slot + 1u); }
    let a = select(0.0, matrixValue(row, previous), previous != 0xffffffffu);
    let c = select(0.0, matrixValue(row, next), next != 0xffffffffu);
    let denominator = max(1e-12, stateF(row, DIAGONAL) - a * previousC);
    let cPrime = c / denominator; let dPrime = (stateF(row, RESIDUAL) - a * previousD) / denominator;
    setStateF(row, MATRIX_DIRECTION, cPrime); setStateF(row, PRECONDITIONED, dPrime);
    previous = row; previousC = cPrime; previousD = dPrime;
  }
  var nextValue = 0.0;
  for (var reverse = end; reverse > first; reverse -= 1u) {
    let row = auxWord(params.precondition.y + reverse - 1u);
    let value = stateF(row, PRECONDITIONED) - stateF(row, MATRIX_DIRECTION) * nextValue;
    setStateF(row, PRECONDITIONED, value); nextValue = value;
  }
}
@compute @workgroup_size(128)
fn preconditionPolynomialStart(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row < dofs() && !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3])) { setStateF(row, PRECONDITIONED, 0.5 * stateF(row, RESIDUAL) / stateF(row, DIAGONAL)); }
}
@compute @workgroup_size(128)
fn preconditionPolynomialMultiply(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  var value = select(0.0, stateF(row, PRECONDITIONED), !dofActive(row));
  if (dofActive(row)) { for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { value += matrixCoefficient(entry) * stateF(matrixNode(entry), PRECONDITIONED); } }
  setStateF(row, MATRIX_DIRECTION, value);
}
@compute @workgroup_size(128)
fn preconditionPolynomialUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < dofs()) { addStateF(row, PRECONDITIONED, 0.5 * (stateF(row, RESIDUAL) - stateF(row, MATRIX_DIRECTION)) / stateF(row, DIAGONAL)); }
}
@compute @workgroup_size(128)
fn startDirection(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < dofs()) { setStateF(gid.x, DIRECTION, stateF(gid.x, PRECONDITIONED)); }
}
fn reducePair(local: u32, a: f32, b: f32) {
  reductionA[local] = a; reductionB[local] = b; workgroupBarrier();
  var stride = 128u;
  loop { if (local < stride) { reductionA[local] += reductionA[local + stride]; reductionB[local] += reductionB[local + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u; }
}
@compute @workgroup_size(256)
fn reduceInitial(@builtin(local_invocation_id) lid: vec3u) {
  var rz = 0.0; var bb = 0.0;
  for (var row = lid.x; row < dofs(); row += 256u) { rz += stateF(row, RESIDUAL) * stateF(row, PRECONDITIONED); bb += stateF(row, RESIDUAL) * stateF(row, RESIDUAL); }
  reducePair(lid.x, rz, bb); if (lid.x == 0u) { scalars[0] = reductionA[0]; scalars[2] = reductionB[0]; scalars[3] = max(reductionB[0], 1e-30); scalars[7] = reductionB[0]; }
}
@compute @workgroup_size(128)
fn multiply(@builtin(global_invocation_id) gid: vec3u) { if (gid.x < dofs()) { setStateF(gid.x, MATRIX_DIRECTION, rowProduct(gid.x)); } }
@compute @workgroup_size(256)
fn applyStep(@builtin(local_invocation_id) lid: vec3u) {
  var value = 0.0; for (var row = lid.x; row < dofs(); row += 256u) { value += stateF(row, DIRECTION) * stateF(row, MATRIX_DIRECTION); }
  reducePair(lid.x, value, 0.0); if (lid.x == 0u) { scalars[1] = reductionA[0]; scalars[4] = select(scalars[0] / max(reductionA[0], 1e-30), 0.0, scalars[2] <= params.solve.x * scalars[3]); }
  storageBarrier(); workgroupBarrier();
  let alpha = scalars[4];
  for (var row = lid.x; row < dofs(); row += 256u) {
    addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION));
    addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION));
  }
}
@compute @workgroup_size(256)
fn finishIteration(@builtin(local_invocation_id) lid: vec3u) {
  var rz = 0.0; var rr = 0.0; for (var row = lid.x; row < dofs(); row += 256u) { rz += stateF(row, RESIDUAL) * stateF(row, PRECONDITIONED); rr += stateF(row, RESIDUAL) * stateF(row, RESIDUAL); }
  reducePair(lid.x, rz, rr); if (lid.x == 0u) { scalars[5] = reductionA[0]; scalars[2] = reductionB[0]; scalars[7] = min(scalars[7], reductionB[0]); scalars[6] = reductionA[0] / max(abs(scalars[0]), 1e-30) * sign(scalars[0]); }
  storageBarrier(); workgroupBarrier();
  for (var row = lid.x; row < dofs(); row += 256u) {
    if (scalars[2] <= scalars[7]) { setStateF(row, BEST_PRESSURE, stateF(row, PRESSURE)); }
    setStateF(row, DIRECTION, stateF(row, PRECONDITIONED) + scalars[6] * stateF(row, DIRECTION));
  }
  if (lid.x == 0u) { scalars[0] = scalars[5]; }
}
fn reducePartial(local: u32, a: f32, b: f32) {
  reductionA[local] = a; reductionB[local] = b; workgroupBarrier();
  var stride = 64u;
  loop { if (local < stride) { reductionA[local] += reductionA[local + stride]; reductionB[local] += reductionB[local + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u; }
}
fn partialBase() -> u32 { return 108u; }
@compute @workgroup_size(128)
fn applyStepPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var value = 0.0; if (gid.x < dofs()) { value = stateF(gid.x, DIRECTION) * stateF(gid.x, MATRIX_DIRECTION); }
  reducePartial(lid.x, value, 0.0); if (lid.x == 0u) { scalars[partialBase() + wid.x] = reductionA[0]; }
}
@compute @workgroup_size(256)
fn applyStepFinalize(@builtin(local_invocation_id) lid: vec3u) {
  var value = 0.0; for (var part = lid.x; part < params.couplingCounts.w; part += 256u) { value += scalars[partialBase() + part]; }
  reducePair(lid.x, value, 0.0); if (lid.x == 0u) { scalars[1] = reductionA[0]; scalars[4] = scalars[0] / max(reductionA[0], 1e-30); }
}
@compute @workgroup_size(128)
fn applyStepUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < dofs()) { let alpha = scalars[4]; addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION)); addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION)); }
}
@compute @workgroup_size(128)
fn finishIterationPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var rz = 0.0; var rr = 0.0;
  if (gid.x < dofs()) { rz = stateF(gid.x, RESIDUAL) * stateF(gid.x, PRECONDITIONED); rr = stateF(gid.x, RESIDUAL) * stateF(gid.x, RESIDUAL); }
  reducePartial(lid.x, rz, rr); if (lid.x == 0u) { scalars[partialBase() + wid.x] = reductionA[0]; scalars[partialBase() + params.couplingCounts.w + wid.x] = reductionB[0]; }
}
@compute @workgroup_size(256)
fn finishIterationFinalize(@builtin(local_invocation_id) lid: vec3u) {
  var rz = 0.0; var rr = 0.0;
  for (var part = lid.x; part < params.couplingCounts.w; part += 256u) { rz += scalars[partialBase() + part]; rr += scalars[partialBase() + params.couplingCounts.w + part]; }
  reducePair(lid.x, rz, rr);
  if (lid.x == 0u) { scalars[5] = reductionA[0]; scalars[2] = reductionB[0]; scalars[7] = min(scalars[7], reductionB[0]); scalars[6] = reductionA[0] / max(abs(scalars[0]), 1e-30) * sign(scalars[0]); scalars[0] = reductionA[0]; }
}
@compute @workgroup_size(128)
fn finishIterationUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < dofs()) { if (scalars[2] <= scalars[7]) { setStateF(row, BEST_PRESSURE, stateF(row, PRESSURE)); } setStateF(row, DIRECTION, stateF(row, PRECONDITIONED) + scalars[6] * stateF(row, DIRECTION)); }
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
    var value = select(stateF(dof, DIRECTION), stateF(dof, BEST_PRESSURE), usePressure);
    if (!dofActive(dof)) { value = 0.0; }
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
  if (dofActive(dof)) { addStateF(dof, MATRIX_DIRECTION, sum); }
}
@compute @workgroup_size(256)
fn coupleImpulse(@builtin(local_invocation_id) lid: vec3u) {
  // Raw K^T p per body; the host converts to impulses via -rho and M^-1.
  for (var body = 0u; body < params.coupling.w; body += 1u) { coupleGather(lid.x, body, true); }
}
fn solvedFaceGradient(face: Face) -> f32 {
  var result = 0.0;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let node = face.nodes[slot]; if (node != 0xffffffffu && faceSlotLiquid(face, slot)) { result += face.coefficients[slot] * stateF(node, BEST_PRESSURE); }
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
    result += bitcast<f32>(auxWord(entries + 2u * cursor + 1u)) * stateF(auxWord(entries + 2u * cursor), BEST_PRESSURE);
  }
  return result;
}
@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; } let id = vec3i(gid);
  var value = textureLoad(velocityIn, id, 0).xyz; let projection = textureLoad(cellProjection, id, 0);
  let ownLiquid = textureLoad(levelSetIn, id, 0).x < 0.0;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var plus = gid; plus[axis] += 1u;
    if (plus[axis] >= params.dims[axis]) { value[axis] = 0.0; continue; }
    let otherLiquid = textureLoad(levelSetIn, vec3i(plus), 0).x < 0.0;
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
  private iterations: number;
  private iterationBudget: QuadtreeIterationBudget;
  private readonly parallelReductions: boolean;
  private readonly phaseQuerySet?: GPUQuerySet;
  private readonly phaseQueryResolve?: GPUBuffer;
  private readonly surfaceState: WebGPUQuadtreeSurfaceState;
  private readonly topologyWords: Uint32Array;
  private lastRelativeResidual?: number;
  levelSetMismatchFraction?: number;
  private readonly couplingBodyCount: number;
  private readonly couplingDistinctDofs: number;
  private readonly couplingBodyIndices: number[];
  private readonly displacedVolumes: number[];
  private lastResidualRms?: number;
  private lastInitialResidualRms?: number;
  private solveSequence = 0;
  private feedbackSequence = -1;

  constructor(private readonly device: GPUDevice, private readonly scene: SceneDescription, private readonly dims: { nx: number; ny: number; nz: number }, private readonly resources: ProjectionResources, private readonly options: QuadtreeTallCellProjectionOptions, fields?: ProjectionFields, private readonly coupling?: QuadtreeRigidCoupling,deferPipelineCompilation=false,cache?:QuadtreeGPUCache) {
    const constructorStartedAt = performance.now();
    const { nx, ny, nz } = dims, h = { x: scene.container.width_m / nx, y: scene.container.height_m / ny, z: scene.container.depth_m / nz };
    const initial: ProjectionFields = fields ?? initialFields(scene, nx, ny, nz);
    if (!fields) initial.phi = signedDistanceFromVolume(Float32Array.from(initial.phi!, (value) => value < 0 ? 1 : 0), nx, ny, nz, h);
    if (!resources.levelSet && !initial.phi) throw new Error("Initial quadtree projection needs a level set");
    // At this point velocityOut still contains the previous step's projected,
    // divergence-free field. velocityIn is the newly advected/gravity-forced
    // predictor and must not transport the resident interface.
    this.surfaceState = resources.levelSet ?? (resources.levelSet = new WebGPUQuadtreeSurfaceState(device, dims, h, resources.velocityOut, initial.phi!, cache?.surface));
    const explicitSizing = initialSizing(scene, nx, nz, h);
    let quadtree = initial.quadtree;
    if (!quadtree) {
      if (!initial.velocity) throw new Error("Initial quadtree construction needs a velocity field");
      if (!initial.phi) throw new Error("Initial quadtree sizing needs a level set");
      const sizing = quadtreeSizingFromVelocityAndSurface(initial.phi, initial.velocity, nx, ny, nz, h);
      for (let index = 0; index < sizing.length; index += 1) sizing[index] = Math.max(sizing[index], explicitSizing[index]);
      quadtree = buildQuadtree(sizing, nx, nz, { h: Math.min(h.x, h.z), maximumLeafSize: options.maximumLeafSize, adaptivityStrength: options.adaptivityStrength, smoothingDilations: 3 });
    }
    const tallGridStartedAt = performance.now();
    if (!initial.pressureGrid && !initial.phi) throw new Error("Initial tall-grid population needs a level set");
    const pressureGrid = initial.pressureGrid ?? populateTallPressureGrid(quadtree, initial.phi!, ny, h, 1, options.opticalDepthFraction);
    this.topologyWords = initial.topologyWords ?? pressureTopologyWords(pressureGrid);
    const tallGrid_ms = initial.prepared?.tallGrid_ms ?? performance.now() - tallGridStartedAt;
    let variationalAssembly_ms: number, systemPack_ms: number;
    let packed: ReturnType<typeof WebGPUQuadtreeTallCellProjection.packSystem>;
    let faceCount: number, ghostFaceCount: number, maximumSystemFluidScale: number, tallSegmentCount: number;
    if (initial.prepared) {
      ({ packed, displacedVolumes: this.displacedVolumes, dofCount: this.dofCount, faceCount, ghostFaceCount, maximumFluidScale: maximumSystemFluidScale, tallSegmentCount, variationalAssembly_ms, systemPack_ms } = initial.prepared);
    } else {
      const variationalStartedAt = performance.now();
      const solidFields = coupling ? solidFieldsFromBodies(scene, coupling.bodies, nx, ny, nz, h) : undefined;
      const variationalBodies = coupling ? variationalBodiesFrom(scene, coupling) : [];
      const system = buildVariationalSystem(pressureGrid, {
        velocity: initial.velocity,
        solidFraction: solidFields?.solidFraction, solidOwner: solidFields?.solidOwner, bodies: variationalBodies
      }, { assembleDense: false });
      variationalAssembly_ms = performance.now() - variationalStartedAt;
      this.displacedVolumes = displacedVolumesForGrid(pressureGrid, initial.phi, solidFields, coupling?.bodies.length ?? 0, nx, ny, h);
      this.dofCount = system.liquidSampleIds.length;
      const packStartedAt = performance.now();
      packed = WebGPUQuadtreeTallCellProjection.packSystem(system, nx, ny, nz, coupling?.dynamic ? variationalBodies : [], options.preconditioner);
      systemPack_ms = performance.now() - packStartedAt;
      faceCount = system.faces.length;
      ghostFaceCount = system.faces.filter((face) => face.ghost).length;
      maximumSystemFluidScale = system.faces.reduce((maximum, face) => Math.max(maximum, face.fluidScale), 0);
      tallSegmentCount = pressureGrid.segments.filter((segment) => segment.tall).length;
    }
    // The hard cap remains the paper-faithful worst case. Only the amount of
    // command stream encoded ahead of the tolerance stop follows recent solves.
    this.iterationBudget = quadtreeIterationBudget(this.dofCount, options);
    this.iterations = this.iterationBudget.encodedBudget;
    this.parallelReductions = this.dofCount >= 4096;
    const uploadStartedAt = performance.now();
    const faces = bufferWithData(device, "Quadtree tall-cell variational faces", packed.faces, GPUBufferUsage.STORAGE, 112);
    const rowOffsets = bufferWithData(device, "Quadtree tall-cell row offsets", packed.rowOffsets);
    const rowEntries = bufferWithData(device, "Quadtree tall-cell row entries", packed.rowEntries, GPUBufferUsage.STORAGE, 8);
    const matrixBuffer = bufferWithData(device, "Quadtree tall-cell refreshed CSR matrix", packed.matrixWords, GPUBufferUsage.STORAGE, 4);
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
    const state = device.createBuffer({ label: "Quadtree tall-cell PCG SoA state", size: Math.max(32, this.dofCount * 32), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // Words 0..11 are the CG scalars; words 12+ hold the per-body coupling
    // six-vectors (stride 8, up to 12 bodies).
    const rowGroups = Math.ceil(this.dofCount / 128), partialWords = 2 * rowGroups;
    const scalars = device.createBuffer({ label: "Quadtree tall-cell CG scalars and partial reductions", size: 4 * (108 + partialWords), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.scalarBuffer = scalars;
    this.dispatchArgs = device.createBuffer({ label: "Quadtree tall-cell active dispatches", size: 76, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.couplingBodyCount = packed.couplingBodyCount; this.couplingDistinctDofs = packed.couplingDistinctDofs; this.couplingBodyIndices = packed.couplingBodyIndices;
    this.params = device.createBuffer({ label: "Quadtree tall-cell parameters", size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([nx, ny, nz, factorAuxWidth, 0, 0, 0, 0]).buffer);
    device.queue.writeBuffer(this.params, 16, new Float32Array([h.x, h.y, h.z, 0]).buffer);
    device.queue.writeBuffer(this.params, 32, new Uint32Array([this.dofCount, faceCount, this.iterations, packed.factorLevelCount]).buffer);
    const solveParams = new ArrayBuffer(16); new Float32Array(solveParams)[0] = options.relativeTolerance ** 2;
    new Uint32Array(solveParams).set([packed.levelsOffset, packed.rowOffsetsOffset, packed.rowEntriesOffset], 1); device.queue.writeBuffer(this.params, 48, solveParams);
    device.queue.writeBuffer(this.params, 64, new Uint32Array([packed.couplingByBodyOffset, packed.couplingByDofOffset, packed.couplingTableOffset, packed.couplingBodyCount]).buffer);
    device.queue.writeBuffer(this.params, 80, new Uint32Array([packed.couplingDistinctDofs, packed.mlsOffsetsBase, packed.mlsRowCount, rowGroups]).buffer);
    device.queue.writeBuffer(this.params, 96, new Uint32Array([packed.lineOffsetsBase, packed.lineDofsBase, packed.lineCount, Math.max(2, Math.min(4, Math.round(options.polynomialDegree ?? 2)))]).buffer);
    const couplingGroups = Math.ceil(packed.couplingDistinctDofs / 128);
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
        ...Array.from({ length: 2 }, (_, index) => ({ binding: index + 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
    this.gpuCache.surface = this.surfaceState.cache;
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
    const names = ["refreshFaces", "refreshRows", "initialize", "precondition", "preconditionJacobi", "preconditionLine", "preconditionPolynomialStart", "preconditionPolynomialMultiply", "preconditionPolynomialUpdate", "startDirection", "reduceInitial", "multiply", "applyStep", "applyStepPartial", "applyStepFinalize", "applyStepUpdate", "finishIteration", "finishIterationPartial", "finishIterationFinalize", "finishIterationUpdate", "project", "coupleReduce", "coupleApply", "coupleImpulse"];
    this.pipelines = this.gpuCache.pipelines ?? (deferPipelineCompilation ? {} : Object.fromEntries(names.map((entryPoint) => [entryPoint, device.createComputePipeline(this.pipelineDescriptor(entryPoint))])));
    if (!this.gpuCache.pipelines && !deferPipelineCompilation) this.gpuCache.pipelines = this.pipelines;
    const all = [faces, rowOffsets, rowEntries, matrixBuffer, state, scalars, factorColumns, factorEntries];
    this.bindGroup = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: resources.velocityIn.createView() }, { binding: 1, resource: resources.velocityOut.createView() }, { binding: 2, resource: resources.volume.createView() },
      ...all.slice(0, 4).map((buffer, index) => ({ binding: index + 3, resource: { buffer } })),
      { binding: 7, resource: this.cellProjection.createView() },
      ...all.slice(4, 8).map((buffer, index) => ({ binding: index + 8, resource: { buffer } })), { binding: 12, resource: { buffer: this.params } },
      { binding: 13, resource: this.factorAux.createView() },
      { binding: 14, resource: this.mlsRowIndex.createView() },
      { binding: 15, resource: this.surfaceState.texture.createView() }
    ] });
    this.buffers = [...all, this.dispatchArgs];
    if (options.debugPressureTimings && device.features.has("timestamp-query")) {
      this.phaseQuerySet = device.createQuerySet({ label: "Quadtree pressure phase timings", type: "timestamp", count: 8 });
      this.phaseQueryResolve = device.createBuffer({ label: "Quadtree pressure phase timing resolve", size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    const allocatedBytes = this.buffers.reduce((sum, buffer) => sum + buffer.size, this.params.size + nx * ny * nz * 24);
    const resourceUpload_ms = performance.now() - uploadStartedAt;
    this.info = { leafCount: quadtree.leaves.length, pressureSampleCount: pressureGrid.samples.length, liquidDofCount: this.dofCount, faceCount, tallSegmentCount, ghostFaceCount, maximumNeighborRatio: quadtree.maximumNeighborRatio, compressionRatio: this.dofCount / Math.max(1, nx * ny * nz), maximumFluidScale: maximumSystemFluidScale, allocatedBytes, cpuTallGrid_ms: tallGrid_ms, cpuVariationalAssembly_ms: variationalAssembly_ms, cpuSystemPack_ms: systemPack_ms, cpuICFactorization_ms: packed.icFactorization_ms, cpuResourceUpload_ms: resourceUpload_ms, cpuTopologyPack_ms: performance.now() - constructorStartedAt, topologyReused: false, topologyReuseCount: 0, pressureIterationBudget: this.iterations, pressureIterationHardBudget: this.iterationBudget.hardBudget, factorLevelCount: packed.factorLevelCount };
  }

  private pipelineDescriptor(entryPoint:string):GPUComputePipelineDescriptor{return{label:`Quadtree tall-cell ${entryPoint}`,layout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  static async createAsync(device:GPUDevice,scene:SceneDescription,dims:{nx:number;ny:number;nz:number},resources:ProjectionResources,options:QuadtreeTallCellProjectionOptions,fields?:ProjectionFields,coupling?:QuadtreeRigidCoupling,onProgress:(label:string,completed:number,total:number)=>void=()=>{},cache?:QuadtreeGPUCache){const projection=new WebGPUQuadtreeTallCellProjection(device,scene,dims,resources,options,fields,coupling,true,cache);await projection.initializePipelines(onProgress);return projection;}
  async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void=()=>{}){
    if(this.gpuCache.pipelines){this.pipelines=this.gpuCache.pipelines;return;}
    const names=["refreshFaces","refreshRows","initialize","precondition","preconditionJacobi","preconditionLine","preconditionPolynomialStart","preconditionPolynomialMultiply","preconditionPolynomialUpdate","startDirection","reduceInitial","multiply","applyStep","applyStepPartial","applyStepFinalize","applyStepUpdate","finishIteration","finishIterationPartial","finishIterationFinalize","finishIterationUpdate","project","coupleReduce","coupleApply","coupleImpulse"];const pipelines:Record<string,GPUComputePipeline>={};for(let index=0;index<names.length;index+=1){const entryPoint=names[index];onProgress(`Adaptive pressure · ${entryPoint}`,index,names.length);pipelines[entryPoint]=await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint));onProgress(`Adaptive pressure · ${entryPoint}`,index+1,names.length);}this.pipelines=pipelines;this.gpuCache.pipelines=pipelines;
  }

  static packSystem(system: VariationalSystem, nx: number, ny: number, nz: number, dynamicBodies: VariationalBody[], preconditioner: QuadtreeTallCellProjectionOptions["preconditioner"] = "ic0") {
    // The last two vec4s retain each pressure sample's leaf origin/y and span.
    // They let a reused topology refresh free-surface weights from the current
    // GPU level set without rebuilding or uploading the sparse face graph.
    // `flux` follows two vec4 fields, so WGSL's 16-byte struct alignment makes
    // the stride 28 words (112 bytes), not 25 words.
    const faceStride = 28, faces = new ArrayBuffer(system.faces.length * faceStride * 4), faceU32 = new Uint32Array(faces), faceF32 = new Float32Array(faces);
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
    // Per-step refresh turns these frozen face contributions into a flat CSR
    // matrix. Each row then performs dense f32/u32 streams with no face-graph
    // gather or four-slot walk inside the CG loop.
    const matrixTerms = incident.map((row) => row.flatMap((item) => {
      const face = system.faces[item.face];
      return face.nodes.flatMap((sample, slot) => {
        const node = system.dofBySample[sample];
        return node < 0 ? [] : [{ node, faceSlot: item.face | (slot << 30), base: item.coefficient * face.coefficients[slot] }];
      });
    }));
    const matrixEntryCount = matrixTerms.reduce((sum, row) => sum + row.length, 0);
    const matrixWords = new Uint32Array(matrixTerms.length + 1 + 4 * matrixEntryCount), matrixFloats = new Float32Array(matrixWords.buffer);
    let matrixEntry = 0;
    matrixTerms.forEach((row, rowIndex) => {
      matrixWords[rowIndex] = matrixEntry;
      for (const item of row) {
        const base = matrixTerms.length + 1 + 4 * matrixEntry;
        matrixWords[base] = item.node; matrixWords[base + 1] = item.faceSlot;
        matrixFloats[base + 2] = 0; matrixFloats[base + 3] = item.base; matrixEntry += 1;
      }
    });
    matrixWords[matrixTerms.length] = matrixEntry;
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const leafId = system.grid.quadtree.leafAt[x + nx * z], leaf = system.grid.quadtree.leaves[leafId];
      for (let y = 0; y < ny; y += 1) {
        const [a, b, wa, wb] = interpolation(system.grid.samplesByLeaf[leafId], y), index = x + nx * (y + ny * z), dofA = system.dofBySample[a], dofB = system.dofBySample[b];
        void dofA; void dofB;
        const representedPhi = wa * system.grid.samples[a].phi + wb * system.grid.samples[b].phi;
        cellProjection[4 * index + 3] = representedPhi < 0 ? leaf.size : -leaf.size;
      }
    }
    const icStartedAt = performance.now();
    // Bridson's public-domain modified incomplete Cholesky level-zero
    // factorization (omega=0.97, minimum pivot ratio=0.25), generalized from
    // the regular seven-point stencil to this sparse variational matrix.
    const n = matrixRows.length, factorStarts = new Uint32Array(n + 1), factorRows: number[] = [], factorValues: number[] = [];
    const factorPositions: Array<Map<number, number>> = Array.from({ length: n }, () => new Map<number, number>());
    const originalDiagonal = new Float64Array(n), workDiagonal = new Float64Array(n), inverseDiagonal = new Float64Array(n);
    const buildIncompleteCholesky = (preconditioner ?? "ic0") === "ic0";
    if (buildIncompleteCholesky) {
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
          // poorly because this variational operator is not an M-matrix. Their
          // selected IC(0) path applies no dropped-fill compensation.
          workDiagonal[targetColumn] -= 0 * multiplier * missing;
        }
      }
    } else factorStarts[n] = 0;
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
    const icFactorization_ms = performance.now() - icStartedAt;
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
    const lineRows = system.grid.samplesByLeaf.map((column) => column.map((sample) => system.dofBySample[sample.id]).filter((dof) => dof >= 0)).filter((row) => row.length > 0);
    const lineOffsetsBase = mlsOffsetsBase + (mlsRows.length + 1) + mlsEntryCount * 2;
    const lineDofsBase = lineOffsetsBase + lineRows.length + 1;
    const lineDofCount = lineRows.reduce((sum, row) => sum + row.length, 0);
    const totalWords = lineDofsBase + lineDofCount;
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
    {
      let cursor = 0;
      lineRows.forEach((row, index) => { factorAuxWords[lineOffsetsBase + index] = cursor; factorAuxWords.set(row, lineDofsBase + cursor); cursor += row.length; });
      factorAuxWords[lineOffsetsBase + lineRows.length] = cursor;
    }
    return {
      faces: new Uint8Array(faces), rowOffsets, rowEntries: new Uint8Array(rowEntries, 0, entryCount * 8), matrixWords, cellProjection, cellTopology,
      factorColumns: new Uint8Array(factorColumnsBuffer), factorEntries: new Uint8Array(factorEntriesBuffer, 0, factorRows.length * 8),
      factorAuxWords, factorLevelCount: levelCount, levelsOffset, rowOffsetsOffset, rowEntriesOffset,
      couplingByBodyOffset, couplingByDofOffset, couplingTableOffset, couplingBodyCount, couplingDistinctDofs,
      couplingBodyIndices: couplings.map((coupling) => coupling.body),
      mlsOffsetsBase, mlsRowCount: mlsRows.length, mlsIndex, icFactorization_ms,
      lineOffsetsBase, lineDofsBase, lineCount: lineRows.length
    };
  }

  encode(encoder: GPUCommandEncoder, nx: number, ny: number, nz: number, timestampWrites?: GPUComputePassTimestampWrites) {
    this.solveSequence += 1;
    encoder.clearBuffer(this.scalarBuffer);
    const coupled = this.couplingBodyCount > 0;
    const rowGroups = Math.ceil(this.dofCount / 128);
    const preconditioner = this.options.preconditioner ?? "ic0";
    const polynomialDegree = Math.max(2, Math.min(4, Math.round(this.options.polynomialDegree ?? 2)));
    const directPrecondition = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void) => {
      if (preconditioner === "ic0") direct("precondition", 1);
      else if (preconditioner === "jacobi") direct("preconditionJacobi", rowGroups);
      else if (preconditioner === "line") direct("preconditionLine", rowGroups);
      else {
        direct("preconditionPolynomialStart", rowGroups);
        for (let degree = 1; degree < polynomialDegree; degree += 1) { direct("preconditionPolynomialMultiply", rowGroups); direct("preconditionPolynomialUpdate", rowGroups); }
      }
    };
    const indirectPrecondition = (indirect: (entry: string, offset: number) => void) => {
      if (preconditioner === "ic0") indirect("precondition", 12);
      else if (preconditioner === "jacobi") indirect("preconditionJacobi", 0);
      else if (preconditioner === "line") indirect("preconditionLine", 0);
      else {
        indirect("preconditionPolynomialStart", 0);
        for (let degree = 1; degree < polynomialDegree; degree += 1) { indirect("preconditionPolynomialMultiply", 0); indirect("preconditionPolynomialUpdate", 0); }
      }
    };
    const withPass = (writes: GPUComputePassTimestampWrites | undefined, encode: (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void, pass: GPUComputePassEncoder) => void) => {
      const pass = encoder.beginComputePass(writes ? { timestampWrites: writes } : undefined);
      const direct = (entry: string, workgroups: number, y = 1, z = 1) => { pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(workgroups, y, z); };
      const indirect = (entry: string, offset: number) => { pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroupsIndirect(this.dispatchArgs, offset); };
      encode(direct, indirect, pass); pass.end();
    };
    const setup = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void) => {
      // Geometry-dependent values are refreshed once, outside the CG loop.
      direct("refreshFaces", Math.ceil(this.info.faceCount / 128)); direct("refreshRows", rowGroups);
      direct("initialize", rowGroups); directPrecondition(direct); direct("startDirection", rowGroups); direct("reduceInitial", 1);
    };
    const iterations = (first: number, end: number, direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void, pass: GPUComputePassEncoder) => {
      void direct;
      for (let iteration = first; iteration < end; iteration += 1) {
        pass.setPipeline(this.dispatchPipeline); pass.setBindGroup(0, this.dispatchBindGroup); pass.dispatchWorkgroups(1);
        indirect("multiply", 0);
        if (coupled) { indirect("coupleReduce", 12); indirect("coupleApply", 24); }
        if (this.parallelReductions) { indirect("applyStepPartial", 0); indirect("applyStepFinalize", 12); indirect("applyStepUpdate", 0); }
        else indirect("applyStep", 12);
        indirectPrecondition(indirect);
        if (this.parallelReductions) { indirect("finishIterationPartial", 0); indirect("finishIterationFinalize", 12); indirect("finishIterationUpdate", 0); }
        else indirect("finishIteration", 12);
      }
    };
    const project = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void) => {
      if (coupled) direct("coupleImpulse", 1);
      direct("project", Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    };
    if (this.phaseQuerySet && this.phaseQueryResolve) {
      if (timestampWrites) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } }); marker.end(); }
      const writes = (beginningOfPassWriteIndex: number, endOfPassWriteIndex: number): GPUComputePassTimestampWrites => ({ querySet: this.phaseQuerySet!, beginningOfPassWriteIndex, endOfPassWriteIndex });
      const firstCount = Math.min(this.iterations, Math.max(1, Math.round(this.options.debugPressureFirstIterations ?? 8)));
      withPass(writes(0, 1), (direct) => setup(direct));
      withPass(writes(2, 3), (direct, indirect, pass) => iterations(0, firstCount, direct, indirect, pass));
      withPass(writes(4, 5), (direct, indirect, pass) => iterations(firstCount, this.iterations, direct, indirect, pass));
      withPass(writes(6, 7), (direct) => project(direct));
      encoder.resolveQuerySet(this.phaseQuerySet, 0, 8, this.phaseQueryResolve, 0);
      if (timestampWrites) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } }); marker.end(); }
    } else {
      // Default path stays in one pass; the debug split is deliberately opt-in.
      withPass(timestampWrites, (direct, indirect, pass) => { setup(direct); iterations(0, this.iterations, direct, indirect, pass); project(direct); });
    }
  }

  encodeSurface(encoder: GPUCommandEncoder, dt_s: number) { this.surfaceState.encode(encoder, dt_s); }
  readSurfaceDiagnostics() { return this.surfaceState.readVolumeDiagnostics(); }
  get surfaceDiagnostics() { return this.surfaceState.volumeDiagnostics; }
  addSurfaceReferenceVolumeCells(cells: number) { this.surfaceState.addReferenceVolumeCells(cells); }

  /** Resident signed-distance field; texture identity is stable across topology rebuilds. */
  get levelSetTexture() { return this.surfaceState.texture; }

  encodeBodyImpulseReadback(encoder: GPUCommandEncoder) {
    if (!this.coupling?.dynamic || this.couplingBodyCount === 0) return undefined;
    const bytes = this.couplingBodyCount * 8 * 4;
    const readback = this.device.createBuffer({ label: "Quadtree per-step rigid impulse readback", size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.scalarBuffer, 12 * 4, readback, 0, bytes);
    return readback;
  }

  async readBodyImpulseReadback(readback: GPUBuffer): Promise<QuadtreeBodyImpulse[]> {
    await readback.mapAsync(GPUMapMode.READ);
    try {
      const solve = new Float32Array(readback.getMappedRange()), rho = this.scene.fluid.density_kg_m3;
      return this.couplingBodyIndices.map((bodyIndex, slot) => {
        const base = slot * 8, body = this.coupling!.bodies[bodyIndex];
        return {
          bodyId: body?.description.id ?? `body-${bodyIndex}`,
          impulse_N_s: { x: -rho * solve[base], y: -rho * solve[base + 1], z: -rho * solve[base + 2] },
          angularImpulse_N_m_s: { x: -rho * solve[base + 3], y: -rho * solve[base + 4], z: -rho * solve[base + 5] },
          displacedVolume_m3: this.displacedVolumes[bodyIndex] ?? 0
        };
      });
    } finally {
      readback.unmap(); readback.destroy();
    }
  }

  async readSolveDiagnostics() {
    // A freshly rebuilt projection may be swapped in after the preceding solve
    // but before stats are sampled. Its scalar buffer is intentionally blank;
    // retain the diagnostics carried over by rebuildFromState until it encodes.
    if (this.solveSequence === 0) return;
    const phaseBytes = this.phaseQueryResolve ? 64 : 0;
    const readback = this.device.createBuffer({ label: "Quadtree solve diagnostics", size: 48 + phaseBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.scalarBuffer, 0, readback, 0, 48);
    if (this.phaseQueryResolve) encoder.copyBufferToBuffer(this.phaseQueryResolve, 0, readback, 48, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const solve = new Float32Array(readback.getMappedRange(0, 48));
      this.applySolveFeedback(solve);
      this.lastRelativeResidual = Math.sqrt(Math.max(0, solve[7]) / Math.max(1e-30, solve[3]));
      this.lastResidualRms = Math.sqrt(Math.max(0, solve[7]) / Math.max(1, this.dofCount));
      this.lastInitialResidualRms = Math.sqrt(Math.max(0, solve[3]) / Math.max(1, this.dofCount));
      if (phaseBytes) {
        const times = new BigUint64Array(readback.getMappedRange(48, 64));
        this.info.pressurePhaseTimings = {
          setup_ms: Number(times[1] - times[0]) / 1e6,
          firstIterations_ms: Number(times[3] - times[2]) / 1e6,
          remainingIterations_ms: Number(times[5] - times[4]) / 1e6,
          project_ms: Number(times[7] - times[6]) / 1e6
        };
      }
    } finally {
      if (readback.mapState === "mapped") readback.unmap(); readback.destroy();
    }
  }

  private applySolveFeedback(solve: Float32Array) {
    const used = Math.round(solve[9] ?? 0);
    const converged = solve[3] > 0 && solve[2] <= this.options.relativeTolerance ** 2 * solve[3];
    this.info.pressureIterationsUsed = used; this.info.pressureConverged = converged;
    if (this.feedbackSequence === this.solveSequence) return;
    this.feedbackSequence = this.solveSequence;
    this.iterationBudget = nextQuadtreeIterationBudget(this.iterationBudget, used, converged);
    this.iterations = this.iterationBudget.encodedBudget;
    this.info.pressureIterationBudget = this.iterations;
    this.info.pressureIterationHardBudget = this.iterationBudget.hardBudget;
    this.device.queue.writeBuffer(this.params, 40, new Uint32Array([this.iterations]));
  }

  private feedbackOptions(): QuadtreeTallCellProjectionOptions {
    return { ...this.options, iterationBudgetHint: this.iterations, iterationEmaHint: this.iterationBudget.ema };
  }

  async rebuildFromState(bodies?: RigidBodyState[]) {
    const { nx, ny, nz } = this.dims;
    const scalarBytes = 48 + 12 * 8 * 4;
    const h = { x: this.scene.container.width_m / nx, y: this.scene.container.height_m / ny, z: this.scene.container.depth_m / nz };
    const activeBodies = bodies ?? this.coupling?.bodies;
    const builder = new WebGPUQuadtreeBuilder(this.device, this.dims, h, this.options.maximumLeafSize, this.options.adaptivityStrength, 3, this.gpuCache.construction);
    this.gpuCache.construction = builder.cache;
    const built = await builder.build({
      velocity: this.resources.velocityOut, volume: this.resources.volume, levelSet: this.surfaceState.texture,
      explicitSizing: initialSizing(this.scene, nx, nz, h, activeBodies), diagnosticBuffer: this.scalarBuffer, diagnosticBytes: scalarBytes
    });
    this.applySolveFeedback(built.diagnostics);
    const nextOptions = this.feedbackOptions();
    const nextCoupling = this.coupling ? { bodies: activeBodies ?? this.coupling.bodies, dynamic: this.coupling.dynamic } : undefined;
    // Decode, tall-cell segmentation, variational assembly, IC factorization,
    // level scheduling, and MLS packing are pure CPU work and run off-main-thread
    // in browsers. Node/test environments use the identical direct fallback.
    const prepared = await prepareQuadtreeProjectionInWorker(this.gpuCache, {
      scene: this.scene, dims: this.dims, options: nextOptions,
      packedCells: built.packedCells, columnProfiles: built.columnProfiles, coupling: nextCoupling
    });
    const reuseTopology = !this.coupling && sameWords(this.topologyWords, prepared.topologyWords);
    let next: WebGPUQuadtreeTallCellProjection;
    if (reuseTopology) {
      const uploadStartedAt = performance.now();
      this.info.cpuTallGrid_ms = prepared.tallGrid_ms;
      this.info.cpuVariationalAssembly_ms = 0;
      this.info.cpuSystemPack_ms = 0;
      this.info.cpuICFactorization_ms = prepared.icFactorization_ms;
      this.info.cpuResourceUpload_ms = performance.now() - uploadStartedAt;
      this.info.topologyReused = true;
      this.info.topologyReuseCount = (this.info.topologyReuseCount ?? 0) + 1;
      // Returning the same projection is the cache-hit signal consumed by the
      // owner; it must not retire the still-resident buffers after the swap.
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      next = this;
    } else {
      const fields: ProjectionFields = { quadtree: prepared.quadtree, pressureGrid: prepared.pressureGrid, topologyWords: prepared.topologyWords, prepared };
      next = this.gpuCache.pipelines
        ? new WebGPUQuadtreeTallCellProjection(this.device, this.scene, this.dims, this.resources, nextOptions, fields, nextCoupling, false, this.gpuCache)
        : await WebGPUQuadtreeTallCellProjection.createAsync(this.device, this.scene, this.dims, this.resources, nextOptions, fields, nextCoupling, undefined, this.gpuCache);
      next.info.cpuTallGrid_ms = prepared.tallGrid_ms;
      next.info.topologyReused = false;
      next.info.topologyReuseCount = this.info.topologyReuseCount ?? 0;
    }
    next.levelSetMismatchFraction = 0;
    next.info.gpuConstruction_ms = built.gpuWall_ms;
    next.info.gpuConstructionKernel_ms = built.gpuKernel_ms;
    next.info.cpuRedistance_ms = 0;
    next.info.cpuQuadtreeDecode_ms = prepared.quadtreeDecode_ms;
    next.info.cpuVariationalAssembly_ms = reuseTopology ? 0 : prepared.variationalAssembly_ms;
    next.info.cpuSystemPack_ms = reuseTopology ? 0 : prepared.systemPack_ms;
    next.info.cpuICFactorization_ms = prepared.icFactorization_ms;
    next.info.cpuTopologyPack_ms = prepared.quadtreeDecode_ms + prepared.tallGrid_ms + prepared.variationalAssembly_ms + prepared.systemPack_ms;
    next.info.topologyReadbackBytes = built.columnProfiles.byteLength + nx * nz * 4 + scalarBytes + 16;
    const solve = built.diagnostics;
    next.info.pressureIterationsUsed = Math.round(solve[9] ?? 0);
    next.info.pressureConverged = solve[3] > 0 && solve[2] <= this.options.relativeTolerance ** 2 * solve[3];
    next.lastRelativeResidual = Math.sqrt(Math.max(0, solve[7]) / Math.max(1e-30, solve[3]));
    next.lastResidualRms = Math.sqrt(Math.max(0, solve[7]) / Math.max(1, this.dofCount));
    next.lastInitialResidualRms = Math.sqrt(Math.max(0, solve[3]) / Math.max(1, this.dofCount));
    return next;
  }

  get relativeResidual() { return this.lastRelativeResidual; }
  get residualRms() { return this.lastResidualRms; }
  get initialResidualRms() { return this.lastInitialResidualRms; }
  get topologyTexture() { return this.cellTopology; }
  get preconditioner() { const value = this.options.preconditioner; return value === "jacobi" || value === "line" || value === "poly" ? value : "ic0"; }

  destroySharedSurface() { this.surfaceState.destroy(); WebGPUQuadtreeBuilder.destroyCache(this.gpuCache.construction); this.gpuCache.cpuWorker?.terminate(); this.gpuCache.cpuWorker = undefined; }
  destroy() { for (const buffer of this.buffers) buffer.destroy(); this.params.destroy(); this.cellProjection.destroy(); this.cellTopology.destroy(); this.factorAux.destroy(); this.mlsRowIndex.destroy(); this.phaseQuerySet?.destroy(); this.phaseQueryResolve?.destroy(); }
}

export function prepareQuadtreeProjectionCPU(input: QuadtreeCPUPreparationInput): PreparedProjectionCPU {
  const { nx, ny, nz } = input.dims;
  const h = { x: input.scene.container.width_m / nx, y: input.scene.container.height_m / ny, z: input.scene.container.depth_m / nz };
  const decodeStartedAt = performance.now();
  const quadtree = quadtreeFromPackedCells(input.packedCells, nx, nz);
  const quadtreeDecode_ms = performance.now() - decodeStartedAt;
  const tallStartedAt = performance.now();
  const pressureGrid = populateTallPressureGridFromLeafProfiles(quadtree, input.columnProfiles, ny, h, input.options.opticalDepthFraction);
  const topologyWords = pressureTopologyWords(pressureGrid);
  const tallGrid_ms = performance.now() - tallStartedAt;
  const assemblyStartedAt = performance.now();
  const solidFields = input.coupling ? solidFieldsFromBodies(input.scene, input.coupling.bodies, nx, ny, nz, h) : undefined;
  const variationalBodies = input.coupling ? variationalBodiesFrom(input.scene, input.coupling) : [];
  const system = buildVariationalSystem(pressureGrid, {
    solidFraction: solidFields?.solidFraction, solidOwner: solidFields?.solidOwner, bodies: variationalBodies
  }, { assembleDense: false });
  const variationalAssembly_ms = performance.now() - assemblyStartedAt;
  const packStartedAt = performance.now();
  const packed = WebGPUQuadtreeTallCellProjection.packSystem(system, nx, ny, nz, input.coupling?.dynamic ? variationalBodies : [], input.options.preconditioner);
  const systemPack_ms = performance.now() - packStartedAt;
  return {
    quadtree, pressureGrid, topologyWords, packed,
    displacedVolumes: displacedVolumesForGrid(pressureGrid, undefined, solidFields, input.coupling?.bodies.length ?? 0, nx, ny, h),
    dofCount: system.liquidSampleIds.length,
    faceCount: system.faces.length,
    ghostFaceCount: system.faces.filter((face) => face.ghost).length,
    maximumFluidScale: system.faces.reduce((maximum, face) => Math.max(maximum, face.fluidScale), 0),
    tallSegmentCount: pressureGrid.segments.filter((segment) => segment.tall).length,
    quadtreeDecode_ms, tallGrid_ms, variationalAssembly_ms, systemPack_ms, icFactorization_ms: packed.icFactorization_ms
  };
}

async function prepareQuadtreeProjectionInWorker(cache: QuadtreeGPUCache, input: QuadtreeCPUPreparationInput) {
  if (!cache.cpuWorker) {
    cache.cpuWorkerSequence = 0; cache.cpuWorkerPending = new Map();
    const receive = (data: { id: number; value?: PreparedProjectionCPU; error?: string }) => {
      const pending = cache.cpuWorkerPending?.get(data.id); if (!pending) return;
      cache.cpuWorkerPending!.delete(data.id);
      if (data.value) pending.resolve(data.value); else pending.reject(new Error(data.error ?? "Quadtree CPU worker failed"));
    };
    const fail = (message: string) => {
      for (const pending of cache.cpuWorkerPending?.values() ?? []) pending.reject(new Error(message));
      cache.cpuWorkerPending?.clear();
    };
    if (typeof Worker !== "undefined") {
      const worker = new Worker(new URL("./quadtree-topology-worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (event: MessageEvent<{ id: number; value?: PreparedProjectionCPU; error?: string }>) => receive(event.data);
      worker.onerror = (event) => fail(event.message);
      cache.cpuWorker = worker;
    } else if (typeof process !== "undefined" && process.versions?.node) {
      // Keep Node smoke timings honest: topology packing must not synchronously
      // stall the driver thread while GPU timings are being sampled.
      const workerThreadsSpecifier = "node:" + "worker_threads";
      const { Worker: NodeWorker } = await import(/* @vite-ignore */ workerThreadsSpecifier) as typeof import("node:worker_threads");
      // Node 25's built-in strip-only loader rejects parameter properties in
      // imported application code. The tiny data-module entry uses tsx's
      // programmatic API, then loads the real worker with full TS transforms.
      const entryUrl = new URL("./quadtree-topology-worker-node.ts", import.meta.url).href;
      const tsxApiUrl = import.meta.resolve("tsx/esm/api");
      const source = `const { tsImport } = await import(${JSON.stringify(tsxApiUrl)}); await tsImport(${JSON.stringify(entryUrl)}, import.meta.url);`;
      const worker = new NodeWorker(new URL(`data:text/javascript,${encodeURIComponent(source)}`), { execArgv: ["--no-strip-types"] });
      worker.on("message", receive); worker.on("error", (error) => fail(error.message));
      cache.cpuWorker = worker;
    } else return prepareQuadtreeProjectionCPU(input);
  }
  const id = (cache.cpuWorkerSequence ?? 0) + 1; cache.cpuWorkerSequence = id;
  return new Promise<PreparedProjectionCPU>((resolve, reject) => {
    cache.cpuWorkerPending!.set(id, { resolve, reject });
    cache.cpuWorker!.postMessage({ id, input });
  });
}
