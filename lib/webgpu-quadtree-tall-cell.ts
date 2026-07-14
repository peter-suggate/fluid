import { advectAndRedistanceLevelSet, buildQuadtree, buildVariationalSystem, populateTallPressureGrid, quadtreeSizingFromVelocityAndSurface, signedDistanceFromVolume, type TallPressureSample, type VariationalSystem } from "./quadtree-tall-cell-grid";
import { damBreakFractions } from "./initial-fluid";
import type { SceneDescription, Vec3 } from "./model";

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
}

interface ProjectionResources {
  velocityIn: GPUTexture;
  velocityOut: GPUTexture;
  volume: GPUTexture;
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

function initialSizing(scene: SceneDescription, phi: ArrayLike<number>, nx: number, ny: number, nz: number, h: Vec3) {
  const sizing = new Float32Array(nx * nz);
  // Interfaces and rigid geometry are explicit sizing sources in the paper's
  // examples. Dynamic velocity/curvature values are refreshed by the GPU grid
  // construction kernels; this initial field prevents progressive refinement
  // from overlooking sub-grid features on the first step.
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y + 1 < ny; y += 1) {
      const a = phi[x + nx * (y + ny * z)] < 0, b = phi[x + nx * (y + 1 + ny * z)] < 0;
      if (a !== b) sizing[x + nx * z] = Math.max(sizing[x + nx * z], 2 / Math.min(h.x, h.z));
    }
    const worldX = -scene.container.width_m / 2 + (x + 0.5) * h.x, worldZ = -scene.container.depth_m / 2 + (z + 0.5) * h.z;
    for (const body of scene.rigidBodies) {
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
struct Params { dims: vec4u, cell: vec4f, counts: vec4u, solve: vec4f }
struct Face { nodes: vec4u, coefficients: vec4f, bounds: vec4u, packed: u32, weights: vec2f }
struct Entry { face: u32, coefficient: f32 }
struct Interpolation { nodes: vec2u, weights: vec2f }
struct SolverState { pressure: f32, bestPressure: f32, residual: f32, direction: f32, preconditioned: f32, matrixDirection: f32, diagonal: f32 }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var volumeIn: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> faces: array<Face>;
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
fn faceVelocity(face: Face) -> f32 {
  let axis = faceAxis(face); let ghost = faceGhost(face);
  if (!ghost) {
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
@compute @workgroup_size(128)
fn initialize(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  var rhs = 0.0; var diag = 0.0;
  for (var entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; entry += 1u) {
    let item = rowEntries[entry]; let face = faces[item.face];
    rhs += item.coefficient * face.weights.x * faceVelocity(face);
    diag += item.coefficient * item.coefficient * face.weights.y;
  }
  state[row].pressure = 0.0; state[row].bestPressure = 0.0; state[row].residual = rhs; state[row].diagonal = max(diag, 1e-12);
  state[row].preconditioned = 0.0; state[row].direction = 0.0; state[row].matrixDirection = 0.0;
}
@compute @workgroup_size(256)
fn precondition(@builtin(local_invocation_id) lid: vec3u) {
  let solveActive = !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3]);
  if (solveActive) { for (var row = lid.x; row < dofs(); row += 256u) { state[row].preconditioned = state[row].residual; } }
  storageBarrier(); workgroupBarrier();
  // The IC graph is level-scheduled so every row within a level is
  // independent. One workgroup supplies the required global ordering while
  // evaluating up to 256 triangular rows concurrently.
  for (var level = 0u; level < params.counts.w; level += 1u) {
    let levelsOffset = bitcast<u32>(params.solve.y); let rowOffsetsOffset = bitcast<u32>(params.solve.z); let rowEntriesOffset = bitcast<u32>(params.solve.w);
    let range = vec2u(auxWord(levelsOffset + 4u * level), auxWord(levelsOffset + 4u * level + 1u));
    if (solveActive) { for (var slot = range.x + lid.x; slot < range.y; slot += 256u) {
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
    if (solveActive) { for (var slot = range.x + lid.x; slot < range.y; slot += 256u) {
      let column = auxWord(slot); var value = state[column].preconditioned;
      for (var entry = factorColumns[column].x; entry < factorColumns[column + 1u].x; entry += 1u) {
        let factor = factorEntries[entry]; value -= factor.coefficient * state[factor.face].preconditioned;
      }
      state[column].preconditioned = value * bitcast<f32>(factorColumns[column].y);
    } }
    storageBarrier(); workgroupBarrier();
  }
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
fn solvedFaceGradient(face: Face) -> f32 {
  var result = 0.0;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let node = face.nodes[slot]; if (node != 0xffffffffu) { result += face.coefficients[slot] * state[node].bestPressure; }
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
      // Eq. (1).  Prolong the same correction to every represented cubic
      // sub-face so its area average remains exactly the solved face value.
      let fluidScale = select(0.0, face.weights.y / face.weights.x, face.weights.x > 0.0);
      value[axis] -= fluidScale * solvedFaceGradient(face);
    }
  }
  textureStore(velocityOut, id, vec4f(value, 0.0));
}
`;

export class WebGPUQuadtreeTallCellProjection {
  readonly info: QuadtreeTallCellProjectionInfo;
  private readonly buffers: GPUBuffer[];
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Record<string, GPUComputePipeline>;
  private readonly params: GPUBuffer;
  private readonly scalarBuffer: GPUBuffer;
  private readonly cellProjection: GPUTexture;
  private readonly factorAux: GPUTexture;
  private readonly dofCount: number;
  private readonly iterations: number;
  private readonly levelSet: Float32Array;
  private lastRelativeResidual?: number;
  private lastResidualRms?: number;
  private lastInitialResidualRms?: number;

  constructor(private readonly device: GPUDevice, private readonly scene: SceneDescription, private readonly dims: { nx: number; ny: number; nz: number }, private readonly resources: ProjectionResources, private readonly options: QuadtreeTallCellProjectionOptions, fields?: { phi: Float32Array; velocity: Vec3[] }) {
    const { nx, ny, nz } = dims, h = { x: scene.container.width_m / nx, y: scene.container.height_m / ny, z: scene.container.depth_m / nz };
    const initial = fields ?? initialFields(scene, nx, ny, nz);
    if (!fields) initial.phi = signedDistanceFromVolume(Float32Array.from(initial.phi, (value) => value < 0 ? 1 : 0), nx, ny, nz, h);
    this.levelSet = Float32Array.from(initial.phi);
    const explicitSizing = initialSizing(scene, initial.phi, nx, ny, nz, h);
    const sizing = quadtreeSizingFromVelocityAndSurface(initial.phi, initial.velocity, nx, ny, nz, h);
    for (let index = 0; index < sizing.length; index += 1) sizing[index] = Math.max(sizing[index], explicitSizing[index]);
    const quadtree = buildQuadtree(sizing, nx, nz, { h: Math.min(h.x, h.z), maximumLeafSize: options.maximumLeafSize, adaptivityStrength: options.adaptivityStrength, smoothingDilations: 3 });
    const opticalDepth = Math.max(1, Math.ceil(ny * scene.container.fillFraction * options.opticalDepthFraction));
    const pressureGrid = populateTallPressureGrid(quadtree, initial.phi, ny, h, opticalDepth);
    const system = buildVariationalSystem(pressureGrid, { velocity: initial.velocity }, { assembleDense: false });
    this.dofCount = system.liquidSampleIds.length; this.iterations = Math.max(1, Math.round(options.pressureIterations));
    const packed = this.packSystem(system, nx, ny, nz);
    const faces = bufferWithData(device, "Quadtree tall-cell variational faces", packed.faces, GPUBufferUsage.STORAGE, 64);
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
    const state = device.createBuffer({ label: "Quadtree tall-cell PCG state", size: Math.max(28, this.dofCount * 28), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const scalars = device.createBuffer({ label: "Quadtree tall-cell CG scalars", size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.scalarBuffer = scalars;
    this.params = device.createBuffer({ label: "Quadtree tall-cell parameters", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([nx, ny, nz, factorAuxWidth, 0, 0, 0, 0]).buffer);
    device.queue.writeBuffer(this.params, 16, new Float32Array([h.x, h.y, h.z, 0]).buffer);
    device.queue.writeBuffer(this.params, 32, new Uint32Array([this.dofCount, system.faces.length, this.iterations, packed.factorLevelCount]).buffer);
    const solveParams = new ArrayBuffer(16); new Float32Array(solveParams)[0] = options.relativeTolerance ** 2;
    new Uint32Array(solveParams).set([packed.levelsOffset, packed.rowOffsetsOffset, packed.rowEntriesOffset], 1); device.queue.writeBuffer(this.params, 48, solveParams);
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      ...Array.from({ length: 4 }, (_, index) => ({ binding: index + 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "2d" } }
    ] });
    const shaderModule = device.createShaderModule({ label: "Quadtree tall-cell variational PCG", code: quadtreeTallCellProjectionShader });
    void shaderModule.getCompilationInfo().then((result) => { for (const message of result.messages) if (message.type === "error") console.error(`Quadtree tall-cell WGSL ${message.lineNum}:${message.linePos} ${message.message}`); });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const names = ["initialize", "precondition", "startDirection", "reduceInitial", "multiply", "reduceDenominator", "updateSolution", "reduceResidual", "saveBest", "updateDirection", "project"];
    this.pipelines = Object.fromEntries(names.map((entryPoint) => [entryPoint, device.createComputePipeline({ label: `Quadtree tall-cell ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } })]));
    const all = [faces, rowOffsets, rowEntries, interpolationBuffer, state, scalars, factorColumns, factorEntries];
    this.bindGroup = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: resources.velocityIn.createView() }, { binding: 1, resource: resources.velocityOut.createView() }, { binding: 2, resource: resources.volume.createView() },
      ...all.slice(0, 4).map((buffer, index) => ({ binding: index + 3, resource: { buffer } })),
      { binding: 7, resource: this.cellProjection.createView() },
      ...all.slice(4, 8).map((buffer, index) => ({ binding: index + 8, resource: { buffer } })), { binding: 12, resource: { buffer: this.params } },
      { binding: 13, resource: this.factorAux.createView() }
    ] });
    this.buffers = all;
    const tallSegmentCount = pressureGrid.segments.filter((segment) => segment.tall).length;
    const allocatedBytes = all.reduce((sum, buffer) => sum + buffer.size, this.params.size + nx * ny * nz * 16);
    this.info = { leafCount: quadtree.leaves.length, pressureSampleCount: pressureGrid.samples.length, liquidDofCount: this.dofCount, faceCount: system.faces.length, tallSegmentCount, ghostFaceCount: system.faces.filter((face) => face.ghost).length, maximumNeighborRatio: quadtree.maximumNeighborRatio, compressionRatio: this.dofCount / Math.max(1, nx * ny * nz), maximumFluidScale: system.faces.reduce((maximum, face) => Math.max(maximum, face.fluidScale), 0), allocatedBytes };
  }

  private packSystem(system: VariationalSystem, nx: number, ny: number, nz: number) {
    // 64-byte face records keep deep-water topology below WebGPU's portable
    // storage-binding limit. The packed word stores span/axis/node count/ghost.
    const faceStride = 16, faces = new ArrayBuffer(system.faces.length * faceStride * 4), faceU32 = new Uint32Array(faces), faceF32 = new Float32Array(faces);
    const incident: Array<Array<{ face: number; coefficient: number }>> = Array.from({ length: system.liquidSampleIds.length }, () => []);
    const cellProjection = new Float32Array(nx * ny * nz * 4);
    const matrixRows: Array<Map<number, number>> = Array.from({ length: system.liquidSampleIds.length }, () => new Map<number, number>());
    system.faces.forEach((face, faceIndex) => {
      const offset = faceIndex * faceStride, nodeCount = face.nodes.length;
      for (let slot = 0; slot < 4; slot += 1) {
        const dof = slot < nodeCount ? system.dofBySample[face.nodes[slot]] : -1;
        faceU32[offset + slot] = dof < 0 ? INVALID : dof;
        faceF32[offset + 4 + slot] = slot < nodeCount ? face.coefficients[slot] : 0;
        if (dof >= 0) incident[dof].push({ face: faceIndex, coefficient: face.coefficients[slot] });
      }
      faceU32.set([face.bounds.x, face.bounds.z, face.bounds.y0, face.bounds.y1], offset + 8);
      faceU32[offset + 12] = face.bounds.span | (face.axis << 16) | (nodeCount << 18) | ((face.ghost ? 1 : 0) << 21);
      const va = face.volume * face.openFraction;
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
    const levelCount = Math.max(1, 1 + Math.max(...forwardLevel, ...backwardLevel));
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
    const factorAuxWords = new Uint32Array(Math.max(4, rowEntriesOffset + 2 * rowEntry));
    factorAuxWords.set(schedule.subarray(0, scheduleOffset), 0); factorAuxWords.set(levels, levelsOffset); factorAuxWords.set(factorRowOffsets, rowOffsetsOffset);
    factorAuxWords.set(new Uint32Array(factorRowEntriesBuffer, 0, 2 * rowEntry), rowEntriesOffset);
    return {
      faces: new Uint8Array(faces), rowOffsets, rowEntries: new Uint8Array(rowEntries, 0, entryCount * 8), interpolation: new Uint8Array(interpolationBuffer), cellProjection,
      factorColumns: new Uint8Array(factorColumnsBuffer), factorEntries: new Uint8Array(factorEntriesBuffer, 0, factorRows.length * 8),
      factorAuxWords, factorLevelCount: levelCount, levelsOffset, rowOffsetsOffset, rowEntriesOffset
    };
  }

  encode(encoder: GPUCommandEncoder, nx: number, ny: number, nz: number) {
    const dispatchRows = (entry: string, workgroups: number) => { const pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(workgroups); pass.end(); };
    encoder.clearBuffer(this.scalarBuffer);
    dispatchRows("initialize", Math.ceil(this.dofCount / 128)); dispatchRows("precondition", 1); dispatchRows("startDirection", Math.ceil(this.dofCount / 128)); dispatchRows("reduceInitial", 1);
    for (let iteration = 0; iteration < this.iterations; iteration += 1) {
      dispatchRows("multiply", Math.ceil(this.dofCount / 128)); dispatchRows("reduceDenominator", 1); dispatchRows("updateSolution", Math.ceil(this.dofCount / 128)); dispatchRows("precondition", 1); dispatchRows("reduceResidual", 1); dispatchRows("saveBest", Math.ceil(this.dofCount / 128)); dispatchRows("updateDirection", Math.ceil(this.dofCount / 128));
    }
    const pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.project); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
  }

  async rebuildFromState(dt_s: number) {
    const { nx, ny, nz } = this.dims, velocityRowBytes = nx * 16, volumeRowBytes = nx * 4;
    const velocityPitch = Math.ceil(velocityRowBytes / 256) * 256, volumePitch = Math.ceil(volumeRowBytes / 256) * 256;
    const velocityReadback = this.device.createBuffer({ label: "Quadtree velocity remesh readback", size: velocityPitch * ny * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const volumeReadback = this.device.createBuffer({ label: "Quadtree volume remesh readback", size: volumePitch * ny * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const scalarReadback = this.device.createBuffer({ label: "Quadtree PCG diagnostic readback", size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Quadtree tall-cell remesh readback" });
    encoder.copyTextureToBuffer({ texture: this.resources.velocityOut }, { buffer: velocityReadback, bytesPerRow: velocityPitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    encoder.copyTextureToBuffer({ texture: this.resources.volume }, { buffer: volumeReadback, bytesPerRow: volumePitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    encoder.copyBufferToBuffer(this.scalarBuffer, 0, scalarReadback, 0, 48);
    this.device.queue.submit([encoder.finish()]);
    await Promise.all([velocityReadback.mapAsync(GPUMapMode.READ), volumeReadback.mapAsync(GPUMapMode.READ), scalarReadback.mapAsync(GPUMapMode.READ)]);
    try {
      const velocityBytes = new Uint8Array(velocityReadback.getMappedRange()), volumeBytes = new Uint8Array(volumeReadback.getMappedRange());
      const velocity: Vec3[] = Array.from({ length: nx * ny * nz }, () => ({ x: 0, y: 0, z: 0 }));
      const volume = new Float32Array(nx * ny * nz), h = { x: this.scene.container.width_m / nx, y: this.scene.container.height_m / ny, z: this.scene.container.depth_m / nz };
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
        const velocityRow = new Float32Array(velocityBytes.buffer, velocityBytes.byteOffset + velocityPitch * (y + ny * z), nx * 4);
        const volumeRow = new Float32Array(volumeBytes.buffer, volumeBytes.byteOffset + volumePitch * (y + ny * z), nx);
        for (let x = 0; x < nx; x += 1) {
          const index = x + nx * (y + ny * z), alpha = Math.max(0, Math.min(1, volumeRow[x]));
          velocity[index] = { x: velocityRow[4 * x], y: velocityRow[4 * x + 1], z: velocityRow[4 * x + 2] };
          // Conservative VOF remains the transported field; after readback it
          // is reinitialized into the coherent signed-distance field required
          // by the paper's sizing and Ando--Batty free-surface weights.
          volume[index] = alpha;
        }
      }
      const phi = advectAndRedistanceLevelSet(this.levelSet, velocity, nx, ny, nz, h, dt_s);
      const next = new WebGPUQuadtreeTallCellProjection(this.device, this.scene, this.dims, this.resources, this.options, { phi, velocity });
      const solve = new Float32Array(scalarReadback.getMappedRange());
      next.lastRelativeResidual = Math.sqrt(Math.max(0, solve[7]) / Math.max(1e-30, solve[3]));
      next.lastResidualRms = Math.sqrt(Math.max(0, solve[7]) / Math.max(1, this.dofCount));
      next.lastInitialResidualRms = Math.sqrt(Math.max(0, solve[3]) / Math.max(1, this.dofCount));
      return next;
    } finally {
      velocityReadback.unmap(); volumeReadback.unmap(); scalarReadback.unmap(); velocityReadback.destroy(); volumeReadback.destroy(); scalarReadback.destroy();
    }
  }

  get relativeResidual() { return this.lastRelativeResidual; }
  get residualRms() { return this.lastResidualRms; }
  get initialResidualRms() { return this.lastInitialResidualRms; }

  destroy() { for (const buffer of this.buffers) buffer.destroy(); this.params.destroy(); this.cellProjection.destroy(); this.factorAux.destroy(); }
}
