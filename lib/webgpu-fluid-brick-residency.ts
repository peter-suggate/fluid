/**
 * GPU-owned logical residency for finest-level fluid bricks.
 *
 * The dense textures remain a compatibility backing store while kernels are
 * migrated to brick payloads.  This page table is nevertheless authoritative
 * for sparse publication and octree work scheduling: only resident core/halo
 * bricks are emitted to the worklist, and dry bricks retire after a short
 * hysteresis window.
 */

export const FLUID_BRICK_RESIDENT = 1;
export const FLUID_BRICK_CORE = 2;
export const FLUID_BRICK_HALO = 4;
export const FLUID_BRICK_ACTIVATED = 8;

export const FLUID_BRICK_WORKLIST_HEADER_WORDS = 16;
export const FLUID_BRICK_ACTIVE_DISPATCH_OFFSET_BYTES = 4;
export const FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES = 20;
export const FLUID_BRICK_TOPOLOGY_DISPATCH_OFFSET_BYTES = 48;

export interface FluidBrickResidencyOptions {
  brickSize?: 4 | 8;
  /** Signed-distance air band retained for advection/interpolation stencils. */
  haloCells?: number;
  /** Consecutive dry publications before a formerly resident brick is freed. */
  retireAfterFrames?: number;
  /** Tree leaf index for every x-major solver brick. */
  leafIndices?: Uint32Array<ArrayBuffer>;
  leafCapacity?: number;
}

export interface FluidBrickResidencyStats {
  resident: number;
  core: number;
  halo: number;
  activated: number;
  retired: number;
  generation: number;
  capacity: number;
}

export interface CPUFluidBrickState {
  flags: number;
  dryFrames: number;
}

export interface CPUFluidBrickClassificationOptions {
  haloPhi: number;
  retireAfterFrames: number;
}

/** Deterministic CPU mirror of the per-brick lifecycle used by unit tests. */
export function classifyCPUFluidBrick(
  minimumPhi: number,
  previous: CPUFluidBrickState = { flags: 0, dryFrames: 0 },
  options: CPUFluidBrickClassificationOptions,
  maximumPhi = minimumPhi,
): CPUFluidBrickState {
  if (!Number.isFinite(minimumPhi) || !Number.isFinite(maximumPhi) || maximumPhi < minimumPhi) {
    throw new RangeError("Brick signed-distance range must be finite and ordered");
  }
  if (!(options.haloPhi >= 0) || !Number.isFinite(options.haloPhi)) throw new RangeError("Brick halo must be finite and non-negative");
  if (!Number.isInteger(options.retireAfterFrames) || options.retireAfterFrames < 0 || options.retireAfterFrames > 0xffff) {
    throw new RangeError("Brick retirement window must be a uint16");
  }
  // A sparse surface band must not retain every negative (deep-liquid)
  // brick. Core pages actually straddle phi=0; halo pages have at least one
  // sample within the requested absolute-distance support.
  const core = minimumPhi <= 0 && maximumPhi >= 0;
  const minimumAbsolutePhi = core ? 0 : Math.min(Math.abs(minimumPhi), Math.abs(maximumPhi));
  const desired = minimumAbsolutePhi < options.haloPhi;
  const wasResident = (previous.flags & FLUID_BRICK_RESIDENT) !== 0;
  const dryFrames = desired ? 0 : Math.min(0xffff, previous.dryFrames + 1);
  const resident = desired || (wasResident && dryFrames <= options.retireAfterFrames);
  const flags = (resident ? FLUID_BRICK_RESIDENT : 0)
    | (core ? FLUID_BRICK_CORE : 0)
    | (resident && !core ? FLUID_BRICK_HALO : 0)
    | (resident && !wasResident ? FLUID_BRICK_ACTIVATED : 0);
  return { flags, dryFrames };
}

export const fluidBrickResidencyShader = /* wgsl */ `
struct Params {
  dimsBrick: vec4u,
  brickDimsCapacity: vec4u,
  settings: vec4f,
}
@group(0) @binding(0) var levelSet: texture_3d<f32>;
// state: low 8 bits flags, high 16 bits consecutive dry publications.
@group(0) @binding(1) var<storage, read_write> states: array<u32>;
// Header words 0..15, active (solver index, leaf index) pairs, then retired pairs.
@group(0) @binding(2) var<storage, read_write> worklist: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> leafIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> leafStates: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

const RESIDENT: u32 = 1u;
const CORE: u32 = 2u;
const HALO: u32 = 4u;
const ACTIVATED: u32 = 8u;
const HEADER_WORDS: u32 = 16u;

fn brickCoordinate(index: u32) -> vec3u {
  let bx = params.brickDimsCapacity.x;
  let by = params.brickDimsCapacity.y;
  return vec3u(index % bx, (index / bx) % by, index / (bx * by));
}

fn tiledDispatch(blocks: u32) -> vec2u {
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  return vec2u(x, y);
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) gid: vec3u) {
  let brickIndex = gid.x;
  let capacity = params.brickDimsCapacity.w;
  if (brickIndex >= capacity || brickIndex >= arrayLength(&states) || brickIndex >= arrayLength(&leafIndices)) { return; }
  let brickSize = params.dimsBrick.w;
  let brick = brickCoordinate(brickIndex);
  let origin = brick * brickSize;
  var minimumPhi = 3.402823e38;
  var maximumPhi = -3.402823e38;
  for (var z = 0u; z < brickSize; z += 1u) {
    for (var y = 0u; y < brickSize; y += 1u) {
      for (var x = 0u; x < brickSize; x += 1u) {
        let cell = origin + vec3u(x, y, z);
        if (all(cell < params.dimsBrick.xyz)) {
          let samplePhi = textureLoad(levelSet, vec3i(cell), 0).x;
          minimumPhi = min(minimumPhi, samplePhi);
          maximumPhi = max(maximumPhi, samplePhi);
        }
      }
    }
  }
  let previous = states[brickIndex];
  let previousFlags = previous & 0xffu;
  let wasResident = (previousFlags & RESIDENT) != 0u;
  let core = minimumPhi <= 0.0 && maximumPhi >= 0.0;
  let minimumAbsolutePhi = select(min(abs(minimumPhi), abs(maximumPhi)), 0.0, core);
  let desired = minimumAbsolutePhi < params.settings.x;
  var dryFrames = select(min(0xffffu, (previous >> 16u) + 1u), 0u, desired);
  let retireAfter = u32(params.settings.y);
  let resident = desired || (wasResident && dryFrames <= retireAfter);
  var flags = select(0u, RESIDENT, resident)
    | select(0u, CORE, core)
    | select(0u, HALO, resident && !core)
    | select(0u, ACTIVATED, resident && !wasResident);
  states[brickIndex] = flags | (dryFrames << 16u);
  let leafIndex = leafIndices[brickIndex];
  if (leafIndex < arrayLength(&leafStates)) { leafStates[leafIndex] = flags; }

  if (resident) {
    let slot = atomicAdd(&worklist[0], 1u);
    if (slot < capacity) {
      let base = HEADER_WORDS + slot * 2u;
      atomicStore(&worklist[base], brickIndex);
      atomicStore(&worklist[base + 1u], leafIndex);
    }
    if (core) { atomicAdd(&worklist[8], 1u); }
    else { atomicAdd(&worklist[9], 1u); }
    if (!wasResident) { atomicAdd(&worklist[10], 1u); }
  } else if (wasResident) {
    let slot = atomicAdd(&worklist[4], 1u);
    if (slot < capacity) {
      let base = HEADER_WORDS + capacity * 2u + slot * 2u;
      atomicStore(&worklist[base], brickIndex);
      atomicStore(&worklist[base + 1u], leafIndex);
    }
    atomicAdd(&worklist[11], 1u);
  }
}

@compute @workgroup_size(1)
fn finalize() {
  let capacity = params.brickDimsCapacity.w;
  let resident = min(atomicLoad(&worklist[0]), capacity);
  let retired = min(atomicLoad(&worklist[4]), capacity);
  let voxelsPerBrick = params.dimsBrick.w * params.dimsBrick.w * params.dimsBrick.w;
  let activeDispatch = tiledDispatch((resident * voxelsPerBrick + 255u) / 256u);
  atomicStore(&worklist[1], activeDispatch.x);
  atomicStore(&worklist[2], activeDispatch.y);
  atomicStore(&worklist[3], 1u);
  let retiredDispatch = tiledDispatch((retired * voxelsPerBrick + 255u) / 256u);
  atomicStore(&worklist[5], retiredDispatch.x);
  atomicStore(&worklist[6], retiredDispatch.y);
  atomicStore(&worklist[7], 1u);
  // Eight 4x4x4 workgroups cover one 8^3 brick. Four-cell bricks use one.
  let topologyGroups = select(1u, 8u, params.dimsBrick.w == 8u);
  let topologyDispatch = tiledDispatch(resident * topologyGroups);
  atomicStore(&worklist[12], topologyDispatch.x);
  atomicStore(&worklist[13], topologyDispatch.y);
  atomicStore(&worklist[14], 1u);
  atomicAdd(&worklist[15], 1u);
}
`;

export class GPUFluidBrickResidency {
  readonly brickSize: 4 | 8;
  readonly brickDimensions: readonly [number, number, number];
  readonly capacity: number;
  readonly leafStates: GPUBuffer;
  readonly worklist: GPUBuffer;
  readonly worklistByteLength: number;
  readonly allocatedBytes: number;

  private readonly device: GPUDevice;
  private readonly states: GPUBuffer;
  private readonly leafIndices: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    dimensions: readonly [number, number, number],
    cellSize: readonly [number, number, number],
    options: FluidBrickResidencyOptions = {},
  ) {
    this.device = device;
    this.brickSize = options.brickSize ?? 8;
    if (this.brickSize !== 4 && this.brickSize !== 8) throw new RangeError("Fluid brick size must be 4 or 8");
    for (const [axis, value] of dimensions.entries()) if (!Number.isInteger(value) || value < 1) throw new RangeError(`Fluid dimension ${axis} must be positive`);
    for (const value of cellSize) if (!(value > 0) || !Number.isFinite(value)) throw new RangeError("Fluid cell size must be positive and finite");
    this.brickDimensions = dimensions.map((value) => Math.ceil(value / this.brickSize)) as [number, number, number];
    this.capacity = this.brickDimensions[0] * this.brickDimensions[1] * this.brickDimensions[2];
    const mapping = options.leafIndices ?? Uint32Array.from({ length: this.capacity }, (_, index) => index);
    if (mapping.length !== this.capacity) throw new RangeError("Fluid brick leaf mapping must cover every solver brick");
    const leafCapacity = options.leafCapacity ?? (mapping.length === 0 ? 1 : Math.max(...mapping) + 1);
    if (!Number.isInteger(leafCapacity) || leafCapacity < 1 || mapping.some((leaf) => leaf >= leafCapacity)) throw new RangeError("Fluid brick leaf capacity is invalid");
    const buffer = (label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBufferView<ArrayBuffer>) => {
      const result = device.createBuffer({ label, size: Math.max(4, size), usage });
      if (data && data.byteLength > 0) device.queue.writeBuffer(result, 0, data);
      return result;
    };
    this.states = buffer("Fluid brick page states", this.capacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const worklistWords = FLUID_BRICK_WORKLIST_HEADER_WORDS + this.capacity * 4;
    this.worklistByteLength = worklistWords * 4;
    this.worklist = buffer("Fluid brick active and retired worklists", worklistWords * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.leafIndices = buffer("Fluid brick to sparse leaf mapping", mapping.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mapping);
    this.leafStates = buffer("Sparse leaf fluid residency", leafCapacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.params = buffer("Fluid brick residency parameters", 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const parameterData = new ArrayBuffer(48), uints = new Uint32Array(parameterData), floats = new Float32Array(parameterData);
    uints.set([dimensions[0], dimensions[1], dimensions[2], this.brickSize], 0);
    uints.set([this.brickDimensions[0], this.brickDimensions[1], this.brickDimensions[2], this.capacity], 4);
    const haloCells = options.haloCells ?? 2;
    const retireAfterFrames = options.retireAfterFrames ?? 3;
    if (!(haloCells >= 0) || !Number.isFinite(haloCells)) throw new RangeError("Fluid brick halo must be finite and non-negative");
    if (!Number.isInteger(retireAfterFrames) || retireAfterFrames < 0 || retireAfterFrames > 0xffff) throw new RangeError("Fluid brick retirement window must be a uint16");
    floats.set([haloCells * Math.max(...cellSize), retireAfterFrames, 0, 0], 8);
    device.queue.writeBuffer(this.params, 0, parameterData);
    const module = device.createShaderModule({ label: "Fluid brick residency shader", code: fluidBrickResidencyShader });
    this.layout = device.createBindGroupLayout({ label: "Fluid brick residency layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.classifyPipeline = device.createComputePipeline({ label: "Classify fluid brick residency", layout: pipelineLayout, compute: { module, entryPoint: "classify" } });
    this.finalizePipeline = device.createComputePipeline({ label: "Finalize fluid brick worklists", layout: pipelineLayout, compute: { module, entryPoint: "finalize" } });
    // The texture binding changes with the projection's ping-pong surface and
    // is therefore created in encode(). Keep the common resources resident.
    this.allocatedBytes = this.capacity * 4 + worklistWords * 4 + mapping.byteLength + leafCapacity * 4 + 48;
  }

  encode(encoder: GPUCommandEncoder, levelSet: GPUTexture): void {
    if (this.destroyed) return;
    // Preserve word 15 as a monotonically increasing GPU generation counter.
    encoder.clearBuffer(this.worklist, 0, (FLUID_BRICK_WORKLIST_HEADER_WORDS - 1) * 4);
    const bindGroup = this.device.createBindGroup({ label: "Fluid brick residency bindings", layout: this.layout, entries: [
      { binding: 0, resource: levelSet.createView() },
      { binding: 1, resource: { buffer: this.states } },
      { binding: 2, resource: { buffer: this.worklist } },
      { binding: 3, resource: { buffer: this.leafIndices } },
      { binding: 4, resource: { buffer: this.leafStates } },
      { binding: 5, resource: { buffer: this.params } },
    ] });
    const classify = encoder.beginComputePass({ label: "Classify evolving fluid bricks" });
    classify.setPipeline(this.classifyPipeline);
    classify.setBindGroup(0, bindGroup);
    classify.dispatchWorkgroups(Math.ceil(this.capacity / 64));
    classify.end();
    const finalize = encoder.beginComputePass({ label: "Finalize evolving fluid brick worklists" });
    finalize.setPipeline(this.finalizePipeline);
    finalize.setBindGroup(0, bindGroup);
    finalize.dispatchWorkgroups(1);
    finalize.end();
  }

  async readStats(): Promise<FluidBrickResidencyStats> {
    if (this.destroyed) return { resident: 0, core: 0, halo: 0, activated: 0, retired: 0, generation: 0, capacity: this.capacity };
    const readback = this.device.createBuffer({ label: "Fluid brick residency readback", size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read fluid brick residency" });
    encoder.copyBufferToBuffer(this.worklist, 0, readback, 0, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange(0, 64));
      return { resident: words[0], retired: words[11], core: words[8], halo: words[9], activated: words[10], generation: words[15], capacity: this.capacity };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.states.destroy();
    this.worklist.destroy();
    this.leafIndices.destroy();
    this.leafStates.destroy();
    this.params.destroy();
  }
}
