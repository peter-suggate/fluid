/**
 * Narita et al. Sec. 4.1 on WebGPU.
 *
 * The expensive update path is deliberately dense and pointer-free on the
 * GPU: one thread first scans each vertical column into a 2D sizing field,
 * then a ping-pong map stores the owning dyadic leaf for every finest x/z
 * cell. Only that one-word-per-column map and the scalar level-set/VOF fields
 * are read back for the still-CPU sparse variational packing stage.
 */

export interface WebGPUQuadtreeConstructionCache {
  layout: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
  shaderModule: GPUShaderModule;
  pipelines: {
    advectLevelSet: GPUComputePipeline;
    seedDistance: GPUComputePipeline;
    jumpFlood: GPUComputePipeline;
    finalizeDistance: GPUComputePipeline;
    evaluateSizing: GPUComputePipeline;
    refine: GPUComputePipeline;
    smoothTopology: GPUComputePipeline;
  };
}

export interface GPUQuadtreeBuildResult {
  advectedPhi: Float32Array;
  packedCells: Uint32Array;
  diagnostics: Float32Array;
  mismatchFraction: number;
  /** Timestamp-query duration of only the adaptive construction kernels. */
  gpuKernel_ms?: number;
  /** Submit-to-map wall time, including older work already queued on the device. */
  gpuWall_ms: number;
}

export const quadtreeConstructionShader = /* wgsl */ `
struct StaticParams { dims: vec4u, cellAndDt: vec4f, sizing: vec4f }
struct PassParams { values: vec4u }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var volumeIn: texture_3d<f32>;
@group(0) @binding(2) var phiIn: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> phiAdvanced: array<f32>;
@group(0) @binding(4) var<storage, read> distanceSeedsIn: array<u32>;
@group(0) @binding(5) var<storage, read> topologyIn: array<u32>;
@group(0) @binding(6) var<storage, read_write> topologyOut: array<u32>;
@group(0) @binding(7) var<uniform> params: StaticParams;
@group(0) @binding(8) var<storage, read> explicitSizing: array<f32>;
@group(0) @binding(9) var<uniform> passParams: PassParams;
@group(0) @binding(10) var<storage, read_write> sizingField: array<f32>;
@group(0) @binding(11) var<storage, read_write> distanceSeedsOut: array<u32>;
@group(0) @binding(12) var<storage, read_write> rebuildStats: array<atomic<u32>>;

fn index2(q: vec2u) -> u32 { return q.x + params.dims.x * q.y; }
fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
fn clamp3(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.dims.xyz) - vec3i(1)); }
fn loadInputPhi(q: vec3i) -> f32 { return textureLoad(phiIn, clamp3(q), 0).x; }
fn loadAdvancedPhi(q: vec3i) -> f32 { return phiAdvanced[index3(vec3u(clamp3(q)))]; }
fn loadVelocity(q: vec3i) -> vec3f { return textureLoad(velocityIn, clamp3(q), 0).xyz; }
fn loadVolume(q: vec3i) -> f32 { return textureLoad(volumeIn, clamp3(q), 0).x; }
fn effectiveWet(q: vec3i) -> bool {
  let p = clamp3(q); let phi = loadAdvancedPhi(p); let wet = loadVolume(p) >= 0.5;
  let decisiveMismatch = (phi < 0.0) != wet && abs(phi) > 0.5 * min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z));
  return select(phi < 0.0, wet, decisiveMismatch);
}
fn trilinearInput(position: vec3f) -> f32 {
  let hi = vec3f(params.dims.xyz - vec3u(1));
  let p = clamp(position, vec3f(0.0), hi);
  let a = vec3i(floor(p)); let b = min(a + vec3i(1), vec3i(params.dims.xyz) - vec3i(1)); let t = fract(p);
  let x00 = mix(loadInputPhi(vec3i(a.x, a.y, a.z)), loadInputPhi(vec3i(b.x, a.y, a.z)), t.x);
  let x10 = mix(loadInputPhi(vec3i(a.x, b.y, a.z)), loadInputPhi(vec3i(b.x, b.y, a.z)), t.x);
  let x01 = mix(loadInputPhi(vec3i(a.x, a.y, b.z)), loadInputPhi(vec3i(b.x, a.y, b.z)), t.x);
  let x11 = mix(loadInputPhi(vec3i(a.x, b.y, b.z)), loadInputPhi(vec3i(b.x, b.y, b.z)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

@compute @workgroup_size(4, 4, 4)
fn advectLevelSet(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let velocity = textureLoad(velocityIn, vec3i(gid), 0).xyz;
  let cellsPerMetre = vec3f(1.0) / params.cellAndDt.xyz;
  let departure = vec3f(gid) - velocity * params.cellAndDt.w * cellsPerMetre;
  phiAdvanced[index3(gid)] = trilinearInput(departure);
}

fn packSeed(q: vec3u) -> u32 { return q.x | (q.y << 10u) | (q.z << 20u); }
fn unpackSeed(word: u32) -> vec3u { return vec3u(word & 1023u, (word >> 10u) & 1023u, (word >> 20u) & 1023u); }
@compute @workgroup_size(4, 4, 4)
fn seedDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let p = vec3i(gid); let wet = effectiveWet(p); let alpha = loadVolume(p); var isInterface = alpha > 0.001 && alpha < 0.999;
  isInterface = isInterface || effectiveWet(p + vec3i(1, 0, 0)) != wet || effectiveWet(p - vec3i(1, 0, 0)) != wet
    || effectiveWet(p + vec3i(0, 1, 0)) != wet || effectiveWet(p - vec3i(0, 1, 0)) != wet
    || effectiveWet(p + vec3i(0, 0, 1)) != wet || effectiveWet(p - vec3i(0, 0, 1)) != wet;
  distanceSeedsOut[index3(gid)] = select(0xffffffffu, packSeed(gid), isInterface);
  let advectedWet = loadAdvancedPhi(p) < 0.0;
  if (advectedWet != (alpha >= 0.5) && abs(loadAdvancedPhi(p)) > 0.5 * min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z))) { atomicAdd(&rebuildStats[0], 1u); }
}
fn seedDistanceSquared(cell: vec3u, word: u32) -> f32 {
  if (word == 0xffffffffu) { return 3.402823e38; }
  let delta = (vec3f(cell) - vec3f(unpackSeed(word))) * params.cellAndDt.xyz;
  return dot(delta, delta);
}
@compute @workgroup_size(4, 4, 4)
fn jumpFlood(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  var best = distanceSeedsIn[index3(gid)]; var bestDistance = seedDistanceSquared(gid, best); let jump = i32(passParams.values.w);
  for (var dz = -1; dz <= 1; dz += 1) { for (var dy = -1; dy <= 1; dy += 1) { for (var dx = -1; dx <= 1; dx += 1) {
    let q = clamp(vec3i(gid) + vec3i(dx, dy, dz) * jump, vec3i(0), vec3i(params.dims.xyz) - vec3i(1));
    let candidate = distanceSeedsIn[index3(vec3u(q))]; let candidateDistance = seedDistanceSquared(gid, candidate);
    if (candidateDistance < bestDistance) { best = candidate; bestDistance = candidateDistance; }
  } } }
  distanceSeedsOut[index3(gid)] = best;
}
@compute @workgroup_size(4, 4, 4)
fn finalizeDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let word = distanceSeedsIn[index3(gid)]; let halfCell = 0.5 * min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z));
  let distance = select(halfCell, sqrt(seedDistanceSquared(gid, word)) + halfCell, word != 0xffffffffu);
  phiAdvanced[index3(gid)] = select(distance, -distance, effectiveWet(vec3i(gid)));
}

@compute @workgroup_size(8, 8)
fn evaluateSizing(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let hx = params.cellAndDt.x; let hy = params.cellAndDt.y; let hz = params.cellAndDt.z;
  let band = 2.0 * max(hx, max(hy, hz)); var maximum = explicitSizing[index2(q)];
  for (var y = 0u; y < params.dims.y; y += 1u) {
    let p = vec3i(i32(q.x), i32(y), i32(q.y)); let phi = loadAdvancedPhi(p); let alpha = loadVolume(p);
    let wet = alpha > 0.5;
    let crosses = (loadVolume(p + vec3i(1, 0, 0)) > 0.5) != wet || (loadVolume(p - vec3i(1, 0, 0)) > 0.5) != wet
      || (loadVolume(p + vec3i(0, 1, 0)) > 0.5) != wet || (loadVolume(p - vec3i(0, 1, 0)) > 0.5) != wet
      || (loadVolume(p + vec3i(0, 0, 1)) > 0.5) != wet || (loadVolume(p - vec3i(0, 0, 1)) > 0.5) != wet;
    if (abs(phi) > band && !(alpha > 0.001 && alpha < 0.999) && !crosses) { continue; }
    let laplacian = (loadAdvancedPhi(p + vec3i(1, 0, 0)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(1, 0, 0))) / (hx * hx)
      + (loadAdvancedPhi(p + vec3i(0, 1, 0)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(0, 1, 0))) / (hy * hy)
      + (loadAdvancedPhi(p + vec3i(0, 0, 1)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(0, 0, 1))) / (hz * hz);
    let vx = (loadVelocity(p + vec3i(1, 0, 0)).x - loadVelocity(p - vec3i(1, 0, 0)).x) / (2.0 * hx);
    let vy = (loadVelocity(p + vec3i(0, 1, 0)).y - loadVelocity(p - vec3i(0, 1, 0)).y) / (2.0 * hy);
    let vz = (loadVelocity(p + vec3i(0, 0, 1)).z - loadVelocity(p - vec3i(0, 0, 1)).z) / (2.0 * hz);
    maximum = max(maximum, params.sizing.x * abs(laplacian) + params.sizing.y * length(vec3f(vx, vy, vz)));
  }
  sizingField[index2(q)] = maximum;
}

fn leafOrigin(word: u32) -> vec2u { return vec2u(word & 1023u, (word >> 10u) & 1023u); }
fn leafSize(word: u32) -> u32 { return (word >> 20u) & 1023u; }
fn packLeaf(origin: vec2u, size: u32) -> u32 { return origin.x | (origin.y << 10u) | (size << 20u); }
fn childForCell(origin: vec2u, size: u32, cell: vec2u) -> u32 {
  let child = size / 2u; var childOrigin = origin;
  if (cell.x >= origin.x + child) { childOrigin.x += child; }
  if (cell.y >= origin.y + child) { childOrigin.y += child; }
  return packLeaf(childOrigin, child);
}

@compute @workgroup_size(8, 8)
fn refine(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let index = index2(q); let word = topologyIn[index]; let size = leafSize(word);
  if (size != passParams.values.x || size <= 1u) { topologyOut[index] = word; return; }
  let origin = leafOrigin(word); var demand = 0.0;
  for (var z = origin.y; z < origin.y + size; z += 1u) { for (var x = origin.x; x < origin.x + size; x += 1u) { demand = max(demand, sizingField[index2(vec2u(x, z))]); } }
  let h = params.sizing.z; let alpha = params.sizing.w; let width = f32(size) * h; var testedWidth = width;
  if (alpha <= 0.0) { testedWidth = h; }
  else if (alpha < 1.0) { testedWidth = exp2(log2(width / h) / log2(1.0 + alpha)) * h; }
  topologyOut[index] = select(word, childForCell(origin, size, q), demand > 1.0 / testedWidth);
}

fn neighborTooFine(origin: vec2u, size: u32, dilation: bool) -> bool {
  for (var offset = 0u; offset < size; offset += 1u) {
    if (origin.x > 0u) { let other = leafSize(topologyIn[index2(vec2u(origin.x - 1u, origin.y + offset))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
    if (origin.x + size < params.dims.x) { let other = leafSize(topologyIn[index2(vec2u(origin.x + size, origin.y + offset))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
    if (origin.y > 0u) { let other = leafSize(topologyIn[index2(vec2u(origin.x + offset, origin.y - 1u))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
    if (origin.y + size < params.dims.z) { let other = leafSize(topologyIn[index2(vec2u(origin.x + offset, origin.y + size))]); if (select(other * 2u < size, other < size, dilation)) { return true; } }
  }
  return false;
}

@compute @workgroup_size(8, 8)
fn smoothTopology(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let index = index2(q); let word = topologyIn[index]; let size = leafSize(word);
  if (size <= 1u) { topologyOut[index] = word; return; }
  let origin = leafOrigin(word); let split = neighborTooFine(origin, size, passParams.values.z != 0u);
  topologyOut[index] = select(word, childForCell(origin, size, q), split);
}
`;

function largestPowerOfTwoAtMost(value: number) {
  let result = 1;
  while (result * 2 <= value) result *= 2;
  return result;
}

/** Same dyadic forest tiling used by the CPU reference builder. */
export function packedQuadtreeRootMap(nx: number, nz: number, maximumLeafSize: number) {
  if (nx <= 0 || nz <= 0 || nx > 1023 || nz > 1023) throw new Error("Unsupported GPU quadtree dimensions");
  const output = new Uint32Array(nx * nz);
  const maximumRootSize = largestPowerOfTwoAtMost(Math.max(1, Math.round(maximumLeafSize)));
  const tile = (x: number, z: number, width: number, depth: number) => {
    if (width <= 0 || depth <= 0) return;
    const size = Math.min(maximumRootSize, largestPowerOfTwoAtMost(Math.min(width, depth)));
    const word = x | (z << 10) | (size << 20);
    for (let zz = z; zz < z + size; zz += 1) for (let xx = x; xx < x + size; xx += 1) output[xx + nx * zz] = word;
    tile(x + size, z, width - size, size);
    tile(x, z + size, width, depth - size);
  };
  tile(0, 0, nx, nz);
  return output;
}

function ensureCache(device: GPUDevice, cache?: WebGPUQuadtreeConstructionCache) {
  if (cache) return cache;
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
  ] });
  const shaderModule = device.createShaderModule({ label: "GPU quadtree construction", code: quadtreeConstructionShader });
  void shaderModule.getCompilationInfo().then((info) => {
    for (const message of info.messages) if (message.type === "error") console.error(`GPU quadtree WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
  }).catch(() => { /* Device loss is reported by the owning solver. */ });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `GPU quadtree ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
  return { layout, shaderModule, pipelineLayout, pipelines: { advectLevelSet: pipeline("advectLevelSet"), seedDistance: pipeline("seedDistance"), jumpFlood: pipeline("jumpFlood"), finalizeDistance: pipeline("finalizeDistance"), evaluateSizing: pipeline("evaluateSizing"), refine: pipeline("refine"), smoothTopology: pipeline("smoothTopology") } };
}

export class WebGPUQuadtreeBuilder {
  readonly cache: WebGPUQuadtreeConstructionCache;
  private readonly rootMap: Uint32Array;

  constructor(
    private readonly device: GPUDevice,
    private readonly dims: { nx: number; ny: number; nz: number },
    private readonly cell: { x: number; y: number; z: number },
    private readonly maximumLeafSize: number,
    private readonly adaptivityStrength: number,
    private readonly smoothingDilations = 3,
    cache?: WebGPUQuadtreeConstructionCache
  ) {
    this.cache = ensureCache(device, cache);
    this.rootMap = packedQuadtreeRootMap(dims.nx, dims.nz, maximumLeafSize);
  }

  async build(inputs: {
    velocity: GPUTexture;
    volume: GPUTexture;
    levelSet: GPUTexture;
    dt_s: number;
    explicitSizing: Float32Array;
    diagnosticBuffer: GPUBuffer;
    diagnosticBytes: number;
  }): Promise<GPUQuadtreeBuildResult> {
    const { nx, ny, nz } = this.dims, cellCount2 = nx * nz;
    if (inputs.explicitSizing.length !== cellCount2) throw new Error("Invalid explicit GPU sizing field");
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const topologyA = this.device.createBuffer({ label: "GPU quadtree topology A", size: Math.max(4, cellCount2 * 4), usage });
    const topologyB = this.device.createBuffer({ label: "GPU quadtree topology B", size: Math.max(4, cellCount2 * 4), usage });
    const sizing = this.device.createBuffer({ label: "GPU quadtree sizing", size: Math.max(4, cellCount2 * 4), usage });
    const explicit = this.device.createBuffer({ label: "GPU quadtree explicit sizing", size: Math.max(4, cellCount2 * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const staticParams = this.device.createBuffer({ label: "GPU quadtree parameters", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const levelSetOut = this.device.createBuffer({ label: "GPU-advected quadtree level set", size: Math.max(4, nx * ny * nz * 4), usage });
    const distanceSeedA = this.device.createBuffer({ label: "GPU quadtree distance seeds A", size: Math.max(4, nx * ny * nz * 4), usage });
    const distanceSeedB = this.device.createBuffer({ label: "GPU quadtree distance seeds B", size: Math.max(4, nx * ny * nz * 4), usage });
    const rebuildStats = this.device.createBuffer({ label: "GPU quadtree reconciliation stats", size: 4, usage });
    const maxRoot = largestPowerOfTwoAtMost(Math.min(Math.max(1, Math.round(this.maximumLeafSize)), nx, nz));
    const balancePasses = Math.max(2, Math.ceil(Math.log2(maxRoot)) + 1);
    const operations: Array<{ entry: "refine" | "smoothTopology"; size: number; mode: number }> = [];
    for (let size = maxRoot; size > 1; size /= 2) operations.push({ entry: "refine", size, mode: 0 });
    for (let pass = 0; pass < balancePasses; pass += 1) operations.push({ entry: "smoothTopology", size: 0, mode: 0 });
    for (let dilation = 0; dilation < this.smoothingDilations; dilation += 1) {
      operations.push({ entry: "smoothTopology", size: 0, mode: 1 });
      for (let pass = 0; pass < balancePasses; pass += 1) operations.push({ entry: "smoothTopology", size: 0, mode: 0 });
    }
    const jumps: number[] = [];
    for (let jump = largestPowerOfTwoAtMost(Math.max(nx, ny, nz)); jump >= 1; jump /= 2) jumps.push(jump);
    const alignment = this.device.limits.minUniformBufferOffsetAlignment;
    const passStride = Math.ceil(16 / alignment) * alignment;
    const passData = new Uint8Array(Math.max(passStride, passStride * (operations.length + jumps.length)));
    operations.forEach((operation, index) => new Uint32Array(passData.buffer, index * passStride, 4).set([operation.size, maxRoot, operation.mode, 0]));
    jumps.forEach((jump, index) => new Uint32Array(passData.buffer, (operations.length + index) * passStride, 4).set([0, 0, 0, jump]));
    const passBuffer = this.device.createBuffer({ label: "GPU quadtree pass parameters", size: passData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(topologyA, 0, this.rootMap.buffer as ArrayBuffer, this.rootMap.byteOffset, this.rootMap.byteLength);
    this.device.queue.writeBuffer(explicit, 0, inputs.explicitSizing.buffer as ArrayBuffer, inputs.explicitSizing.byteOffset, inputs.explicitSizing.byteLength);
    const staticData = new ArrayBuffer(48);
    new Uint32Array(staticData, 0, 4).set([nx, ny, nz, 0]);
    new Float32Array(staticData, 16, 4).set([this.cell.x, this.cell.y, this.cell.z, inputs.dt_s]);
    new Float32Array(staticData, 32, 4).set([4, 3, Math.min(this.cell.x, this.cell.z), Math.max(0, Math.min(1, this.adaptivityStrength))]);
    this.device.queue.writeBuffer(staticParams, 0, staticData);
    this.device.queue.writeBuffer(passBuffer, 0, passData);
    this.device.queue.writeBuffer(rebuildStats, 0, new Uint32Array([0]));

    const group = (input: GPUBuffer, output: GPUBuffer, seedInput: GPUBuffer, seedOutput: GPUBuffer) => this.device.createBindGroup({ layout: this.cache.layout, entries: [
      { binding: 0, resource: inputs.velocity.createView() }, { binding: 1, resource: inputs.volume.createView() },
      { binding: 2, resource: inputs.levelSet.createView() }, { binding: 3, resource: { buffer: levelSetOut } },
      { binding: 4, resource: { buffer: seedInput } },
      { binding: 5, resource: { buffer: input } }, { binding: 6, resource: { buffer: output } },
      { binding: 7, resource: { buffer: staticParams } }, { binding: 8, resource: { buffer: explicit } },
      { binding: 9, resource: { buffer: passBuffer, size: 16 } }, { binding: 10, resource: { buffer: sizing } },
      { binding: 11, resource: { buffer: seedOutput } }, { binding: 12, resource: { buffer: rebuildStats } }
    ] });
    const groupAB = group(topologyA, topologyB, distanceSeedA, distanceSeedB), groupBA = group(topologyB, topologyA, distanceSeedA, distanceSeedB);
    const seedAB = group(topologyA, topologyB, distanceSeedA, distanceSeedB), seedBA = group(topologyA, topologyB, distanceSeedB, distanceSeedA);
    // One mapped staging buffer is the only GPU/CPU boundary in a rebuild.
    // The old path mapped phi, VOF, topology, diagnostics, stats, and timing
    // independently even though VOF was never consumed by the sparse packer.
    const align = (value: number, alignment: number) => Math.ceil(value / alignment) * alignment;
    const phiBytes = nx * ny * nz * 4, topologyBytes = cellCount2 * 4;
    const phiOffset = 0, topologyOffset = align(phiOffset + phiBytes, 8);
    const diagnosticOffset = align(topologyOffset + topologyBytes, 8);
    const statsOffset = align(diagnosticOffset + inputs.diagnosticBytes, 8);
    const queryOffset = align(statsOffset + 4, 8);
    const querySet = this.device.features.has("timestamp-query") ? this.device.createQuerySet({ type: "timestamp", count: 2 }) : undefined;
    const queryResolve = querySet ? this.device.createBuffer({ label: "GPU quadtree timestamp resolve", size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : undefined;
    const readbackBytes = queryOffset + (querySet ? 16 : 0);
    const readback = this.device.createBuffer({ label: "GPU quadtree compact readback", size: Math.max(8, readbackBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "GPU quadtree construction" });
    {
      const pass = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0 } } : undefined); pass.setPipeline(this.cache.pipelines.advectLevelSet); pass.setBindGroup(0, groupAB, [0]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    }
    {
      const pass = encoder.beginComputePass(); pass.setPipeline(this.cache.pipelines.seedDistance); pass.setBindGroup(0, seedAB, [0]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    }
    let currentSeedIsA = false;
    jumps.forEach((_, index) => {
      const pass = encoder.beginComputePass(); pass.setPipeline(this.cache.pipelines.jumpFlood); pass.setBindGroup(0, currentSeedIsA ? seedAB : seedBA, [(operations.length + index) * passStride]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end(); currentSeedIsA = !currentSeedIsA;
    });
    {
      const pass = encoder.beginComputePass(); pass.setPipeline(this.cache.pipelines.finalizeDistance); pass.setBindGroup(0, currentSeedIsA ? seedAB : seedBA, [0]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    }
    {
      const pass = encoder.beginComputePass(); pass.setPipeline(this.cache.pipelines.evaluateSizing); pass.setBindGroup(0, groupAB, [0]);
      pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(nz / 8)); pass.end();
    }
    let currentIsA = true;
    operations.forEach((operation, index) => {
      const last = index === operations.length - 1;
      const pass = encoder.beginComputePass(last && querySet ? { timestampWrites: { querySet, endOfPassWriteIndex: 1 } } : undefined); pass.setPipeline(this.cache.pipelines[operation.entry]); pass.setBindGroup(0, currentIsA ? groupAB : groupBA, [index * passStride]);
      pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(nz / 8)); pass.end(); currentIsA = !currentIsA;
    });
    const finalTopology = currentIsA ? topologyA : topologyB;
    encoder.copyBufferToBuffer(levelSetOut, 0, readback, phiOffset, phiBytes);
    encoder.copyBufferToBuffer(finalTopology, 0, readback, topologyOffset, topologyBytes);
    encoder.copyBufferToBuffer(inputs.diagnosticBuffer, 0, readback, diagnosticOffset, inputs.diagnosticBytes);
    encoder.copyBufferToBuffer(rebuildStats, 0, readback, statsOffset, 4);
    if (querySet && queryResolve) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
      encoder.copyBufferToBuffer(queryResolve, 0, readback, queryOffset, 16);
    }
    const submittedAt = performance.now(); this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    try {
      const mapped = readback.getMappedRange();
      const advectedPhi = new Float32Array(mapped, phiOffset, phiBytes / 4).slice();
      const packedCells = new Uint32Array(mapped, topologyOffset, topologyBytes / 4).slice();
      const diagnostics = new Float32Array(mapped, diagnosticOffset, inputs.diagnosticBytes / 4).slice();
      const mismatchFraction = new Uint32Array(mapped, statsOffset, 1)[0] / Math.max(1, nx * ny * nz);
      const timestamps = querySet ? new BigUint64Array(mapped, queryOffset, 2) : undefined;
      const gpuKernel_ms = timestamps ? Number(timestamps[1] - timestamps[0]) / 1e6 : undefined;
      return { advectedPhi, packedCells, diagnostics, mismatchFraction, gpuKernel_ms, gpuWall_ms: performance.now() - submittedAt };
    } finally {
      readback.unmap(); querySet?.destroy();
      for (const buffer of [topologyA, topologyB, sizing, explicit, staticParams, passBuffer, levelSetOut, distanceSeedA, distanceSeedB, rebuildStats, readback, queryResolve]) buffer?.destroy();
    }
  }
}
