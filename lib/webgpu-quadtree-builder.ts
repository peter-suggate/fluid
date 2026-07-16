/**
 * Narita et al. Sec. 4.1 on WebGPU.
 *
 * The expensive update path is deliberately dense and pointer-free on the
 * GPU: one thread first scans each vertical column into a 2D sizing field,
 * then a ping-pong map stores the owning dyadic leaf for every finest x/z
 * cell. The resident level set is advanced independently every simulation
 * step; construction only evaluates sizing and rebuilds the horizontal tree.
 */

export interface WebGPUQuadtreeSurfaceCache {
  layout: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
  shaderModule: GPUShaderModule;
  pipelines: {
    advectLevelSet: GPUComputePipeline;
    reduceVolume: GPUComputePipeline;
    seedDistance: GPUComputePipeline;
    jumpFlood: GPUComputePipeline;
    finalizeDistance: GPUComputePipeline;
  };
}

/**
 * Per-step level-set transport used by the adaptive pressure path.
 *
 * Narita Alg. 1 advects phi once per physics step. Keeping a canonical GPU
 * texture plus a scratch texture makes that cadence independent of the much
 * slower CPU sparse-topology rebuild. The canonical texture never changes
 * identity, so every projection rebuilt from the shared state sees the newest
 * surface without a CPU upload.
 */
export const quadtreeSurfaceShader = /* wgsl */ `
struct SurfaceParams { dims: vec4u, cellAndDt: vec4f, control: vec4f }
struct PassParams { jump: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var phiIn: texture_3d<f32>;
@group(0) @binding(2) var phiOut: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<storage, read> distanceSeedsIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> distanceSeedsOut: array<u32>;
@group(0) @binding(5) var<uniform> params: SurfaceParams;
@group(0) @binding(6) var<uniform> passParams: PassParams;
@group(0) @binding(7) var<storage, read_write> reductions: array<atomic<u32>>;

fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
fn clamp3(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.dims.xyz) - vec3i(1)); }
fn loadPhi(q: vec3i) -> f32 { return textureLoad(phiIn, clamp3(q), 0).x; }
fn loadVelocity(q: vec3i) -> vec3f { return textureLoad(velocityIn, clamp3(q), 0).xyz; }
fn centredMacVelocity(q: vec3i) -> vec3f {
  let own = loadVelocity(q);
  // velocityIn stores the negative-face MAC sample for each component. The
  // level-set trace needs the cell-centre velocity (Narita Alg. 1 line 4).
  return 0.5 * vec3f(
    own.x + loadVelocity(q - vec3i(1, 0, 0)).x,
    own.y + loadVelocity(q - vec3i(0, 1, 0)).y,
    own.z + loadVelocity(q - vec3i(0, 0, 1)).z
  );
}
fn trilinearPhi(position: vec3f) -> f32 {
  let hi = vec3f(params.dims.xyz - vec3u(1));
  let p = clamp(position, vec3f(0.0), hi);
  let a = vec3i(floor(p)); let b = min(a + vec3i(1), vec3i(params.dims.xyz) - vec3i(1)); let t = fract(p);
  let x00 = mix(loadPhi(vec3i(a.x, a.y, a.z)), loadPhi(vec3i(b.x, a.y, a.z)), t.x);
  let x10 = mix(loadPhi(vec3i(a.x, b.y, a.z)), loadPhi(vec3i(b.x, b.y, a.z)), t.x);
  let x01 = mix(loadPhi(vec3i(a.x, a.y, b.z)), loadPhi(vec3i(b.x, a.y, b.z)), t.x);
  let x11 = mix(loadPhi(vec3i(a.x, b.y, b.z)), loadPhi(vec3i(b.x, b.y, b.z)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn hMin() -> f32 { return min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z)); }
fn volumeCorrectedPhi(value: f32) -> f32 {
  let h = hMin();
  return select(value, value - params.control.x * h * params.cellAndDt.w, abs(value) < 1.5 * h);
}
@compute @workgroup_size(4, 4, 4)
fn advectLevelSet(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  // Narita et al. Sec. 4.5: interpolate velocity from the saved previous
  // staggered grid, backtrace, then interpolate the previous level set.
  let departure = vec3f(gid) - centredMacVelocity(vec3i(gid)) * params.cellAndDt.w / params.cellAndDt.xyz;
  textureStore(phiOut, vec3i(gid), vec4f(volumeCorrectedPhi(trilinearPhi(departure)), 0.0, 0.0, 0.0));
}
@compute @workgroup_size(4, 4, 4)
fn reduceVolume(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let value = loadPhi(vec3i(gid)); let cell = hMin();
  // Match the renderer and smoke oracle: adaptive phi is converted to a
  // smooth Heaviside over four vertical cell widths.
  let occupied = clamp(0.5 - value / (4.0 * params.cellAndDt.y), 0.0, 1.0);
  atomicAdd(&reductions[0], u32(occupied * 256.0));
  if (abs(value) < 1.5 * cell) { atomicAdd(&reductions[1], 256u); }
}
fn packSeed(q: vec3u) -> u32 { return q.x | (q.y << 10u) | (q.z << 20u); }
fn unpackSeed(word: u32) -> vec3u { return vec3u(word & 1023u, (word >> 10u) & 1023u, (word >> 20u) & 1023u); }
@compute @workgroup_size(4, 4, 4)
fn seedDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let p = vec3i(gid); let wet = loadPhi(p) < 0.0;
  let crosses = (loadPhi(p + vec3i(1, 0, 0)) < 0.0) != wet || (loadPhi(p - vec3i(1, 0, 0)) < 0.0) != wet
    || (loadPhi(p + vec3i(0, 1, 0)) < 0.0) != wet || (loadPhi(p - vec3i(0, 1, 0)) < 0.0) != wet
    || (loadPhi(p + vec3i(0, 0, 1)) < 0.0) != wet || (loadPhi(p - vec3i(0, 0, 1)) < 0.0) != wet;
  distanceSeedsOut[index3(gid)] = select(0xffffffffu, packSeed(gid), crosses);
}
fn seedDistanceSquared(cell: vec3u, word: u32) -> f32 {
  if (word == 0xffffffffu) { return 3.402823e38; }
  let delta = (vec3f(cell) - vec3f(unpackSeed(word))) * params.cellAndDt.xyz;
  return dot(delta, delta);
}
@compute @workgroup_size(4, 4, 4)
fn jumpFlood(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  var best = distanceSeedsIn[index3(gid)]; var bestDistance = seedDistanceSquared(gid, best); let jump = i32(passParams.jump);
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
  let word = distanceSeedsIn[index3(gid)];
  let h = min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z)); var distance = 5.0 * h;
  if (word != 0xffffffffu) {
    distance = min(5.0 * h, sqrt(seedDistanceSquared(gid, word)) + 0.5 * h);
  }
  textureStore(phiOut, vec3i(gid), vec4f(select(distance, -distance, loadPhi(vec3i(gid)) < 0.0), 0.0, 0.0, 0.0));
}
`;

function uploadLevelSetTexture(device: GPUDevice, texture: GPUTexture, phi: Float32Array, nx: number, ny: number, nz: number) {
  const rowBytes = nx * 4, pitch = Math.ceil(rowBytes / 256) * 256;
  const upload = new Uint8Array(pitch * ny * nz), source = new Uint8Array(phi.buffer, phi.byteOffset, phi.byteLength);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) upload.set(source.subarray(rowBytes * (y + ny * z), rowBytes * (y + ny * z + 1)), pitch * (y + ny * z));
  device.queue.writeTexture({ texture }, upload, { bytesPerRow: pitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
}

function ensureSurfaceCache(device: GPUDevice, cache?: WebGPUQuadtreeSurfaceCache) {
  if (cache) return cache;
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
  ] });
  const shaderModule = device.createShaderModule({ label: "Resident quadtree level set", code: quadtreeSurfaceShader });
  void shaderModule.getCompilationInfo().then((info) => {
    for (const message of info.messages) if (message.type === "error") console.error(`Resident quadtree level-set WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
  }).catch(() => { /* Device loss is reported by the owning solver. */ });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `Quadtree surface ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
  return { layout, pipelineLayout, shaderModule, pipelines: {
    advectLevelSet: pipeline("advectLevelSet"), reduceVolume: pipeline("reduceVolume"),
    seedDistance: pipeline("seedDistance"), jumpFlood: pipeline("jumpFlood"), finalizeDistance: pipeline("finalizeDistance")
  } };
}

export class WebGPUQuadtreeSurfaceState {
  readonly cache: WebGPUQuadtreeSurfaceCache;
  readonly texture: GPUTexture;
  private readonly scratch: GPUTexture;
  private readonly seedsA: GPUBuffer;
  private readonly seedsB: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly passBuffer: GPUBuffer;
  private readonly reductions: GPUBuffer;
  private readonly passStride: number;
  private readonly jumps: number[];
  private readonly groups: { advect: GPUBindGroup; reduce: GPUBindGroup; seed: GPUBindGroup; jumpAB: GPUBindGroup; jumpBA: GPUBindGroup; finalizeA: GPUBindGroup; finalizeB: GPUBindGroup };
  private readbackPending = false;
  private referenceVolumeCells: number;
  private volumeCells: number;
  private interfaceCells = 0;
  private correctionSpeed = 0;

  constructor(private readonly device: GPUDevice, private readonly dims: { nx: number; ny: number; nz: number }, private readonly cell: { x: number; y: number; z: number }, velocity: GPUTexture, initialPhi: Float32Array, cache?: WebGPUQuadtreeSurfaceCache) {
    this.cache = ensureSurfaceCache(device, cache);
    const textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
    this.texture = device.createTexture({ label: "Resident quadtree level set", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    this.scratch = device.createTexture({ label: "Resident quadtree level-set advection scratch", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "r32float", usage: textureUsage });
    uploadLevelSetTexture(device, this.texture, initialPhi, dims.nx, dims.ny, dims.nz);
    const bytes = Math.max(4, dims.nx * dims.ny * dims.nz * 4), seedUsage = GPUBufferUsage.STORAGE;
    this.seedsA = device.createBuffer({ label: "Quadtree surface seeds A", size: bytes, usage: seedUsage });
    this.seedsB = device.createBuffer({ label: "Quadtree surface seeds B", size: bytes, usage: seedUsage });
    this.params = device.createBuffer({ label: "Quadtree surface parameters", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductions = device.createBuffer({ label: "Quadtree level-set volume diagnostics", size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const volumeBand = 4 * cell.y;
    this.referenceVolumeCells = initialPhi.reduce((sum, value) => sum + Math.max(0, Math.min(1, 0.5 - value / volumeBand)), 0);
    this.volumeCells = this.referenceVolumeCells;
    this.jumps = [];
    for (let jump = largestPowerOfTwoAtMost(Math.max(dims.nx, dims.ny, dims.nz)); jump >= 1; jump /= 2) this.jumps.push(jump);
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    this.passStride = Math.ceil(16 / alignment) * alignment;
    const passData = new Uint8Array(Math.max(this.passStride, this.passStride * this.jumps.length));
    this.jumps.forEach((jump, index) => new Uint32Array(passData.buffer, index * this.passStride, 4).set([jump, 0, 0, 0]));
    this.passBuffer = device.createBuffer({ label: "Quadtree surface pass parameters", size: passData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.passBuffer, 0, passData);
    const group = (phiIn: GPUTexture, phiOut: GPUTexture, seedIn: GPUBuffer, seedOut: GPUBuffer) => device.createBindGroup({ layout: this.cache.layout, entries: [
      { binding: 0, resource: velocity.createView() }, { binding: 1, resource: phiIn.createView() }, { binding: 2, resource: phiOut.createView() },
      { binding: 3, resource: { buffer: seedIn } }, { binding: 4, resource: { buffer: seedOut } }, { binding: 5, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: this.passBuffer, size: 16 } }, { binding: 7, resource: { buffer: this.reductions } }
    ] });
    this.groups = {
      advect: group(this.texture, this.scratch, this.seedsA, this.seedsB),
      reduce: group(this.texture, this.scratch, this.seedsA, this.seedsB),
      seed: group(this.scratch, this.texture, this.seedsB, this.seedsA),
      jumpAB: group(this.scratch, this.texture, this.seedsA, this.seedsB),
      jumpBA: group(this.scratch, this.texture, this.seedsB, this.seedsA),
      finalizeA: group(this.scratch, this.texture, this.seedsA, this.seedsB),
      finalizeB: group(this.scratch, this.texture, this.seedsB, this.seedsA)
    };
  }

  encode(encoder: GPUCommandEncoder, dt_s: number) {
    const { nx, ny, nz } = this.dims;
    const parameterData = new ArrayBuffer(48);
    new Uint32Array(parameterData, 0, 4).set([nx, ny, nz, 0]);
    new Float32Array(parameterData, 16, 4).set([this.cell.x, this.cell.y, this.cell.z, dt_s]);
    new Float32Array(parameterData, 32, 4).set([this.correctionSpeed, 0, 0, 0]);
    this.device.queue.writeBuffer(this.params, 0, parameterData);
    const dispatch = (pipeline: GPUComputePipeline, group: GPUBindGroup, offset = 0) => {
      const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group, [offset]);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    };
    dispatch(this.cache.pipelines.advectLevelSet, this.groups.advect);
    encoder.copyTextureToTexture({ texture: this.scratch }, { texture: this.texture }, { width: nx, height: ny, depthOrArrayLayers: nz });
    encoder.clearBuffer(this.reductions);
    dispatch(this.cache.pipelines.reduceVolume, this.groups.reduce);
  }

  async readVolumeDiagnostics() {
    if (this.readbackPending) return this.volumeDiagnostics;
    this.readbackPending = true;
    const readback = this.device.createBuffer({ label: "Quadtree level-set volume readback", size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.reductions, 0, readback, 0, 8); this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()); this.volumeCells = words[0] / 256; this.interfaceCells = words[1] / 256;
      // The smooth Heaviside is four cells wide, so its derivative converts a
      // one-cell normal shift into roughly one quarter cell of measured
      // volume. Correct one measured error over the renderer's 30 Hz
      // diagnostic interval; the regular solver's half-error gain was too
      // gentle for this four-cell band during the dam-break release.
      this.correctionSpeed = Math.max(-30, Math.min(30, 4 * (this.referenceVolumeCells - this.volumeCells) / Math.max(this.interfaceCells, 1) / (1 / 30)));
    } finally {
      if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); this.readbackPending = false;
    }
    return this.volumeDiagnostics;
  }

  get volumeDiagnostics() { return { referenceVolumeCells: this.referenceVolumeCells, volumeCells: this.volumeCells, interfaceCells: this.interfaceCells, correctionSpeed: this.correctionSpeed }; }
  addReferenceVolumeCells(cells: number) { if (Number.isFinite(cells) && cells > 0) this.referenceVolumeCells += cells; }

  destroy() {
    this.texture.destroy(); this.scratch.destroy(); this.seedsA.destroy(); this.seedsB.destroy(); this.params.destroy(); this.passBuffer.destroy(); this.reductions.destroy();
  }
}

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
    sampleLeafProfiles: GPUComputePipeline;
  };
  workspace?: {
    key: string;
    topologyA: GPUBuffer; topologyB: GPUBuffer; sizing: GPUBuffer; explicit: GPUBuffer;
    staticParams: GPUBuffer; levelSetOut: GPUBuffer; distanceSeedA: GPUBuffer; distanceSeedB: GPUBuffer;
    passBuffer: GPUBuffer; readback: GPUBuffer;
    columnProfiles: GPUBuffer; profileReadback: GPUBuffer;
    querySet?: GPUQuerySet; queryResolve?: GPUBuffer;
  };
}

export interface GPUQuadtreeBuildResult {
  /** Phi sampled at each decoded leaf's horizontal centre, leaf-major. */
  columnProfiles: Float32Array;
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
@group(0) @binding(13) var<storage, read_write> columnProfiles: array<f32>;

fn index2(q: vec2u) -> u32 { return q.x + params.dims.x * q.y; }
fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
fn clamp3(q: vec3i) -> vec3i { return clamp(q, vec3i(0), vec3i(params.dims.xyz) - vec3i(1)); }
fn loadInputPhi(q: vec3i) -> f32 { return textureLoad(phiIn, clamp3(q), 0).x; }
fn loadAdvancedPhi(q: vec3i) -> f32 { return phiAdvanced[index3(vec3u(clamp3(q)))]; }
fn loadVelocity(q: vec3i) -> vec3f { return textureLoad(velocityIn, clamp3(q), 0).xyz; }
fn loadVolume(q: vec3i) -> f32 { return textureLoad(volumeIn, clamp3(q), 0).x; }
fn effectiveWet(q: vec3i) -> bool {
  return loadAdvancedPhi(clamp3(q)) < 0.0;
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
  let q = vec3i(gid); let own = loadVelocity(q);
  let velocity = 0.5 * vec3f(
    own.x + loadVelocity(q - vec3i(1, 0, 0)).x,
    own.y + loadVelocity(q - vec3i(0, 1, 0)).y,
    own.z + loadVelocity(q - vec3i(0, 0, 1)).z
  );
  let cellsPerMetre = vec3f(1.0) / params.cellAndDt.xyz;
  let departure = vec3f(gid) - velocity * params.cellAndDt.w * cellsPerMetre;
  phiAdvanced[index3(gid)] = trilinearInput(departure);
}

fn packSeed(q: vec3u) -> u32 { return q.x | (q.y << 10u) | (q.z << 20u); }
fn unpackSeed(word: u32) -> vec3u { return vec3u(word & 1023u, (word >> 10u) & 1023u, (word >> 20u) & 1023u); }
@compute @workgroup_size(4, 4, 4)
fn seedDistance(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let p = vec3i(gid); let wet = effectiveWet(p); var isInterface = effectiveWet(p + vec3i(1, 0, 0)) != wet || effectiveWet(p - vec3i(1, 0, 0)) != wet
    || effectiveWet(p + vec3i(0, 1, 0)) != wet || effectiveWet(p - vec3i(0, 1, 0)) != wet
    || effectiveWet(p + vec3i(0, 0, 1)) != wet || effectiveWet(p - vec3i(0, 0, 1)) != wet;
  distanceSeedsOut[index3(gid)] = select(0xffffffffu, packSeed(gid), isInterface);
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
  let word = distanceSeedsIn[index3(gid)]; let hMin = min(params.cellAndDt.x, min(params.cellAndDt.y, params.cellAndDt.z)); var distance = 5.0 * hMin;
  if (word != 0xffffffffu) {
    distance = min(5.0 * hMin, sqrt(seedDistanceSquared(gid, word)) + 0.5 * hMin);
  }
  phiAdvanced[index3(gid)] = select(distance, -distance, effectiveWet(vec3i(gid)));
}

fn strainMagnitude(q: vec3i) -> f32 {
  let h = params.cellAndDt.xyz;
  let vx = (loadVelocity(q + vec3i(1, 0, 0)).x - loadVelocity(q - vec3i(1, 0, 0)).x) / (2.0 * h.x);
  let vy = (loadVelocity(q + vec3i(0, 1, 0)).y - loadVelocity(q - vec3i(0, 1, 0)).y) / (2.0 * h.y);
  let vz = (loadVelocity(q + vec3i(0, 0, 1)).z - loadVelocity(q - vec3i(0, 0, 1)).z) / (2.0 * h.z);
  return length(vec3f(vx, vy, vz));
}

@compute @workgroup_size(8, 8)
fn evaluateSizing(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let hx = params.cellAndDt.x; let hy = params.cellAndDt.y; let hz = params.cellAndDt.z;
  let band = 2.0 * max(hx, max(hy, hz)); var maximum = explicitSizing[index2(q)];
  for (var y = 0u; y < params.dims.y; y += 1u) {
    let p = vec3i(i32(q.x), i32(y), i32(q.y)); let phi = loadAdvancedPhi(p); let wet = phi < 0.0;
    let crosses = effectiveWet(p + vec3i(1, 0, 0)) != wet || effectiveWet(p - vec3i(1, 0, 0)) != wet
      || effectiveWet(p + vec3i(0, 1, 0)) != wet || effectiveWet(p - vec3i(0, 1, 0)) != wet
      || effectiveWet(p + vec3i(0, 0, 1)) != wet || effectiveWet(p - vec3i(0, 0, 1)) != wet;
    if (abs(phi) > band && !crosses) { continue; }
    let laplacian = (loadAdvancedPhi(p + vec3i(1, 0, 0)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(1, 0, 0))) / (hx * hx)
      + (loadAdvancedPhi(p + vec3i(0, 1, 0)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(0, 1, 0))) / (hy * hy)
      + (loadAdvancedPhi(p + vec3i(0, 0, 1)) - 2.0 * phi + loadAdvancedPhi(p - vec3i(0, 0, 1))) / (hz * hz);
    maximum = max(maximum, params.sizing.x * abs(laplacian) + params.sizing.y * strainMagnitude(p));
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

@compute @workgroup_size(64)
fn sampleLeafProfiles(@builtin(global_invocation_id) gid: vec3u) {
  let leaf = gid.x; if (leaf >= passParams.values.x) { return; }
  let word = bitcast<u32>(explicitSizing[leaf]);
  let origin = leafOrigin(word); let size = leafSize(word);
  let position = vec2f(origin) + vec2f(f32(size) * 0.5 - 0.5);
  let a = vec2u(floor(position)); let b = min(a + vec2u(1), params.dims.xz - vec2u(1)); let t = fract(position);
  for (var y = 0u; y < params.dims.y; y += 1u) {
    let p00 = loadInputPhi(vec3i(i32(a.x), i32(y), i32(a.y)));
    let p10 = loadInputPhi(vec3i(i32(b.x), i32(y), i32(a.y)));
    let p01 = loadInputPhi(vec3i(i32(a.x), i32(y), i32(b.y)));
    let p11 = loadInputPhi(vec3i(i32(b.x), i32(y), i32(b.y)));
    columnProfiles[leaf * params.dims.y + y] = mix(mix(p00, p10, t.x), mix(p01, p11, t.x), t.y);
  }
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
    { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
  ] });
  const shaderModule = device.createShaderModule({ label: "GPU quadtree construction", code: quadtreeConstructionShader });
  void shaderModule.getCompilationInfo().then((info) => {
    for (const message of info.messages) if (message.type === "error") console.error(`GPU quadtree WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
  }).catch(() => { /* Device loss is reported by the owning solver. */ });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `GPU quadtree ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
  return { layout, shaderModule, pipelineLayout, pipelines: { advectLevelSet: pipeline("advectLevelSet"), seedDistance: pipeline("seedDistance"), jumpFlood: pipeline("jumpFlood"), finalizeDistance: pipeline("finalizeDistance"), evaluateSizing: pipeline("evaluateSizing"), refine: pipeline("refine"), smoothTopology: pipeline("smoothTopology"), sampleLeafProfiles: pipeline("sampleLeafProfiles") } };
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
    explicitSizing: Float32Array;
    diagnosticBuffer: GPUBuffer;
    diagnosticBytes: number;
  }): Promise<GPUQuadtreeBuildResult> {
    const { nx, ny, nz } = this.dims, cellCount2 = nx * nz;
    if (inputs.explicitSizing.length !== cellCount2) throw new Error("Invalid explicit GPU sizing field");
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
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
    const align = (value: number, alignment: number) => Math.ceil(value / alignment) * alignment;
    const phiBytes = nx * ny * nz * 4, topologyBytes = cellCount2 * 4;
    const topologyOffset = 0;
    const diagnosticOffset = align(topologyOffset + topologyBytes, 8);
    const queryOffset = align(diagnosticOffset + inputs.diagnosticBytes, 8);
    const hasTimestamps = this.device.features.has("timestamp-query");
    const readbackBytes = queryOffset + (hasTimestamps ? 16 : 0);
    const workspaceKey = `${nx}x${ny}x${nz}:${passData.byteLength}:${readbackBytes}`;
    if (this.cache.workspace?.key !== workspaceKey) {
      WebGPUQuadtreeBuilder.destroyWorkspace(this.cache.workspace);
      this.cache.workspace = {
        key: workspaceKey,
        topologyA: this.device.createBuffer({ label: "GPU quadtree topology A", size: Math.max(4, topologyBytes), usage }),
        topologyB: this.device.createBuffer({ label: "GPU quadtree topology B", size: Math.max(4, topologyBytes), usage }),
        sizing: this.device.createBuffer({ label: "GPU quadtree sizing", size: Math.max(4, topologyBytes), usage }),
        explicit: this.device.createBuffer({ label: "GPU quadtree explicit sizing", size: Math.max(4, topologyBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
        staticParams: this.device.createBuffer({ label: "GPU quadtree parameters", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        levelSetOut: this.device.createBuffer({ label: "GPU quadtree level-set readback source", size: Math.max(4, phiBytes), usage }),
        distanceSeedA: this.device.createBuffer({ label: "GPU quadtree binding scratch A", size: 4, usage }),
        distanceSeedB: this.device.createBuffer({ label: "GPU quadtree binding scratch B", size: 4, usage }),
        passBuffer: this.device.createBuffer({ label: "GPU quadtree pass parameters", size: passData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        readback: this.device.createBuffer({ label: "GPU quadtree compact readback", size: Math.max(8, readbackBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        columnProfiles: this.device.createBuffer({ label: "GPU quadtree leaf-centre phi profiles", size: Math.max(4, phiBytes), usage }),
        profileReadback: this.device.createBuffer({ label: "GPU quadtree phi-profile readback", size: Math.max(4, phiBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        querySet: hasTimestamps ? this.device.createQuerySet({ type: "timestamp", count: 2 }) : undefined,
        queryResolve: hasTimestamps ? this.device.createBuffer({ label: "GPU quadtree timestamp resolve", size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : undefined
      };
    }
    const { topologyA, topologyB, sizing, explicit, staticParams, levelSetOut, distanceSeedA, distanceSeedB, passBuffer, readback, columnProfiles, profileReadback, querySet, queryResolve } = this.cache.workspace!;
    this.device.queue.writeBuffer(topologyA, 0, this.rootMap.buffer as ArrayBuffer, this.rootMap.byteOffset, this.rootMap.byteLength);
    this.device.queue.writeBuffer(explicit, 0, inputs.explicitSizing.buffer as ArrayBuffer, inputs.explicitSizing.byteOffset, inputs.explicitSizing.byteLength);
    const staticData = new ArrayBuffer(48);
    new Uint32Array(staticData, 0, 4).set([nx, ny, nz, 0]);
    // Construction consumes the already-advanced resident level set. A zero-dt
    // trace only supplies the dense storage buffer used by sizing; no
    // accumulated-time advection occurs during rebuilds.
    new Float32Array(staticData, 16, 4).set([this.cell.x, this.cell.y, this.cell.z, 0]);
    new Float32Array(staticData, 32, 4).set([4, 3, Math.min(this.cell.x, this.cell.z), Math.max(0, Math.min(1, this.adaptivityStrength))]);
    this.device.queue.writeBuffer(staticParams, 0, staticData);
    this.device.queue.writeBuffer(passBuffer, 0, passData);

    const group = (input: GPUBuffer, output: GPUBuffer, seedInput: GPUBuffer, seedOutput: GPUBuffer) => this.device.createBindGroup({ layout: this.cache.layout, entries: [
      { binding: 0, resource: inputs.velocity.createView() }, { binding: 1, resource: inputs.volume.createView() },
      { binding: 2, resource: inputs.levelSet.createView() }, { binding: 3, resource: { buffer: levelSetOut } },
      { binding: 4, resource: { buffer: seedInput } },
      { binding: 5, resource: { buffer: input } }, { binding: 6, resource: { buffer: output } },
      { binding: 7, resource: { buffer: staticParams } }, { binding: 8, resource: { buffer: explicit } },
      { binding: 9, resource: { buffer: passBuffer, size: 16 } }, { binding: 10, resource: { buffer: sizing } },
      { binding: 11, resource: { buffer: seedOutput } },
      { binding: 13, resource: { buffer: columnProfiles } }
    ] });
    const groupAB = group(topologyA, topologyB, distanceSeedA, distanceSeedB), groupBA = group(topologyB, topologyA, distanceSeedA, distanceSeedB);
    // First read back the compact leaf-owner map. Once decoded, a second tiny
    // GPU pass samples only one vertical phi profile per unique leaf instead
    // of transferring the complete dense 3D level set.
    const encoder = this.device.createCommandEncoder({ label: "GPU quadtree construction" });
    {
      const pass = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0 } } : undefined); pass.setPipeline(this.cache.pipelines.advectLevelSet); pass.setBindGroup(0, groupAB, [0]);
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
    encoder.copyBufferToBuffer(finalTopology, 0, readback, topologyOffset, topologyBytes);
    encoder.copyBufferToBuffer(inputs.diagnosticBuffer, 0, readback, diagnosticOffset, inputs.diagnosticBytes);
    if (querySet && queryResolve) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
      encoder.copyBufferToBuffer(queryResolve, 0, readback, queryOffset, 16);
    }
    const submittedAt = performance.now(); this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    let packedCells: Uint32Array, diagnostics: Float32Array, gpuKernel_ms: number | undefined;
    try {
      const mapped = readback.getMappedRange();
      packedCells = new Uint32Array(mapped, topologyOffset, topologyBytes / 4).slice();
      diagnostics = new Float32Array(mapped, diagnosticOffset, inputs.diagnosticBytes / 4).slice();
      const timestamps = querySet ? new BigUint64Array(mapped, queryOffset, 2) : undefined;
      gpuKernel_ms = timestamps ? Number(timestamps[1] - timestamps[0]) / 1e6 : undefined;
    } finally {
      readback.unmap();
    }
    const leafWords = Uint32Array.from(new Set(packedCells));
    const profileBytes = leafWords.length * ny * 4;
    this.device.queue.writeBuffer(explicit, 0, leafWords);
    this.device.queue.writeBuffer(passBuffer, 0, new Uint32Array([leafWords.length, 0, 0, 0]));
    const profileEncoder = this.device.createCommandEncoder({ label: "GPU quadtree leaf profiles" });
    {
      const pass = profileEncoder.beginComputePass(); pass.setPipeline(this.cache.pipelines.sampleLeafProfiles); pass.setBindGroup(0, groupAB, [0]);
      pass.dispatchWorkgroups(Math.ceil(leafWords.length / 64)); pass.end();
    }
    profileEncoder.copyBufferToBuffer(columnProfiles, 0, profileReadback, 0, profileBytes);
    this.device.queue.submit([profileEncoder.finish()]);
    await profileReadback.mapAsync(GPUMapMode.READ, 0, profileBytes);
    let profiles: Float32Array;
    try { profiles = new Float32Array(profileReadback.getMappedRange(0, profileBytes)).slice(); }
    finally { profileReadback.unmap(); }
    return { columnProfiles: profiles, packedCells, diagnostics, mismatchFraction: 0, gpuKernel_ms, gpuWall_ms: performance.now() - submittedAt };
  }

  private static destroyWorkspace(workspace?: WebGPUQuadtreeConstructionCache["workspace"]) {
    if (!workspace) return;
    for (const buffer of [workspace.topologyA, workspace.topologyB, workspace.sizing, workspace.explicit, workspace.staticParams, workspace.levelSetOut, workspace.distanceSeedA, workspace.distanceSeedB, workspace.passBuffer, workspace.readback, workspace.columnProfiles, workspace.profileReadback, workspace.queryResolve]) buffer?.destroy();
    workspace.querySet?.destroy();
  }

  static destroyCache(cache?: WebGPUQuadtreeConstructionCache) {
    WebGPUQuadtreeBuilder.destroyWorkspace(cache?.workspace);
    if (cache) cache.workspace = undefined;
  }
}
