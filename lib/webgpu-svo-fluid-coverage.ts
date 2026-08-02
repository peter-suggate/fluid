import {
  packSvoFluidCoverageFrame,
  planSvoFluidCoverageVolume,
  SVO_FLUID_COVERAGE_LAYOUT,
  type SvoFluidCoveragePlan,
  type SvoFluidCoveragePlanOptions,
  type SvoFluidCoverageTriple,
} from "./svo-fluid-coverage";

/**
 * Per-frame fluid coverage volume, resampled from the solver's coarse level set.
 *
 * The coarse field is a dense r32float signed distance over the container box
 * that the solver rewrites every step and the water pipeline already samples
 * (`globalCoarsePhi`). It owns sign and distance everywhere the fine narrow band
 * does not, which is exactly the question a shadow asks, so the fill is a
 * resample rather than a traversal: no octree descent, no page directory, no
 * residency gate.
 *
 * Reads are `textureLoad` only. r32float is not filterable, so the source binds
 * as unfilterable-float and any smoothing happens in this volume's own mips.
 */

export const SVO_FLUID_COVERAGE_BINDINGS = Object.freeze({
  coarsePhi: 0,
  params: 1,
  destination: 2,
} as const);

export interface WebGpuSvoFluidCoverageSources {
  /** Dense coarse level set, signed distance in metres, negative inside water. */
  coarsePhi: GPUTextureView;
}

export type WebGpuSvoFluidCoverageOptions = SvoFluidCoveragePlanOptions;

export interface WebGpuSvoFluidCoverageGeneration {
  view: GPUTextureView;
  sampler: GPUSampler;
  plan: SvoFluidCoveragePlan;
  /** Increments on every encoded fill; zero means nothing has been written yet. */
  generation: number;
}

const FILL_PARAM_WORDS = 12;
const REDUCE_PARAM_WORDS = 8;
const UNIFORM_ALIGNMENT = 256;

/** Exported so shader compilation is testable without constructing the owner. */
export function svoFluidCoverageFillShader(cellsPerTexel: number): string {
  return /* wgsl */ `
// Twelve words exactly: vec3u members align to 16, so cellDiagonal_m lands at
// word 8 and the struct rounds to 48 bytes. A trailing padding member would
// push it to 64 and silently invalidate a 48-byte uniform binding.
struct FillParams {
  dimensions: vec3u,
  _padding0: u32,
  fieldDimensions: vec3u,
  _padding1: u32,
  cellDiagonal_m: f32,
}
@group(0) @binding(${SVO_FLUID_COVERAGE_BINDINGS.coarsePhi}) var coarsePhi: texture_3d<f32>;
@group(0) @binding(${SVO_FLUID_COVERAGE_BINDINGS.params}) var<uniform> params: FillParams;
@group(0) @binding(${SVO_FLUID_COVERAGE_BINDINGS.destination}) var destination: texture_storage_3d<${SVO_FLUID_COVERAGE_LAYOUT.format}, write>;

// Cells outside the field contribute zero rather than clamping to the boundary
// value, which would extend the edge cell's water along every axis. Only the
// trailing texel of a non-divisible axis is affected, and it borders air.
fn coverageAtCell(cell: vec3u) -> f32 {
  if (any(cell >= params.fieldDimensions)) { return 0.0; }
  let signedDistance = textureLoad(coarsePhi, vec3i(cell), 0).x;
  if (signedDistance != signedDistance) { return 0.0; }
  return clamp(0.5 - signedDistance / params.cellDiagonal_m, 0.0, 1.0);
}

@compute @workgroup_size(${SVO_FLUID_COVERAGE_LAYOUT.fillWorkgroupSize}, ${SVO_FLUID_COVERAGE_LAYOUT.fillWorkgroupSize}, ${SVO_FLUID_COVERAGE_LAYOUT.fillWorkgroupSize})
fn fillFluidCoverage(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dimensions)) { return; }
  let baseCell = gid * ${cellsPerTexel}u;
  var total = 0.0;
  for (var z = 0u; z < ${cellsPerTexel}u; z += 1u) {
    for (var y = 0u; y < ${cellsPerTexel}u; y += 1u) {
      for (var x = 0u; x < ${cellsPerTexel}u; x += 1u) {
        total += coverageAtCell(baseCell + vec3u(x, y, z));
      }
    }
  }
  textureStore(destination, vec3i(gid), vec4f(total / ${(cellsPerTexel ** 3).toFixed(1)}, 0.0, 0.0, 1.0));
}
`;
}

export const svoFluidCoverageReduceShader = /* wgsl */ `
struct ReduceParams { destinationDimensions: vec3u, _padding0: u32, sourceDimensions: vec3u, _padding1: u32 }
@group(0) @binding(0) var source: texture_3d<f32>;
@group(0) @binding(1) var destination: texture_storage_3d<${SVO_FLUID_COVERAGE_LAYOUT.format}, write>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

// Mean of eight, mirroring reduceSvoFluidCoverage. An odd source axis clamps
// its trailing sample rather than averaging fewer, which keeps the reduction
// total-preserving in the interior and only doubles one boundary texel.
@compute @workgroup_size(${SVO_FLUID_COVERAGE_LAYOUT.reduceWorkgroupSize}, ${SVO_FLUID_COVERAGE_LAYOUT.reduceWorkgroupSize}, ${SVO_FLUID_COVERAGE_LAYOUT.reduceWorkgroupSize})
fn reduceFluidCoverage(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.destinationDimensions)) { return; }
  var total = 0.0;
  for (var index = 0u; index < 8u; index += 1u) {
    let offset = vec3u(index & 1u, (index >> 1u) & 1u, (index >> 2u) & 1u);
    let coordinate = min(gid * 2u + offset, params.sourceDimensions - vec3u(1u));
    total += textureLoad(source, vec3i(coordinate), 0).x;
  }
  textureStore(destination, vec3i(gid), vec4f(total / 8.0, 0.0, 0.0, 1.0));
}
`;

function levelDimensions(dimensions: SvoFluidCoverageTriple, level: number): [number, number, number] {
  return dimensions.map((component) => Math.max(1, component >> level)) as [number, number, number];
}

function dispatchCounts(dimensions: readonly number[], workgroup: number): [number, number, number] {
  return dimensions.map((component) => Math.ceil(component / workgroup)) as [number, number, number];
}

/**
 * GPU owner for the dynamic fluid coverage volume.
 *
 * Allocation is deferred to the first `encode`, so a dry fluid-free scene — which
 * shares this octree implementation but never runs a solver step — pays nothing.
 */
export class WebGpuSvoFluidCoverage {
  readonly plan: SvoFluidCoveragePlan;
  private readonly sampler: GPUSampler;
  private readonly cellDiagonal_m: number;
  private texture?: GPUTexture;
  private sampledView?: GPUTextureView;
  private fillPipeline?: GPUComputePipeline;
  private reducePipeline?: GPUComputePipeline;
  private fillLayout?: GPUBindGroupLayout;
  private fillBindGroup?: GPUBindGroup;
  private reduceBindGroups: GPUBindGroup[] = [];
  private uniforms?: GPUBuffer;
  private generation = 0;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    options: WebGpuSvoFluidCoverageOptions,
    private readonly sources: WebGpuSvoFluidCoverageSources,
  ) {
    this.plan = planSvoFluidCoverageVolume(options);
    this.cellDiagonal_m = Math.hypot(...options.cellSize_m);
    this.sampler = device.createSampler({
      label: "SVO fluid coverage sampler",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge",
      magFilter: "linear",
      minFilter: "linear",
      // Linear mip filtering is what makes the cone's level transition
      // continuous. The paged node-mip atlas cannot do this — its levels are
      // separate pages, so it hand-blends two fetches inside a narrow band.
      mipmapFilter: "linear",
    });
  }

  get allocatedBytes(): number {
    return this.texture ? this.plan.allocatedBytes : 0;
  }

  /** Undefined until at least one fill has been encoded, so the renderer fails closed to no water. */
  visibleGeneration(): WebGpuSvoFluidCoverageGeneration | undefined {
    if (this.destroyed || !this.sampledView || this.generation === 0) return undefined;
    return { view: this.sampledView, sampler: this.sampler, plan: this.plan, generation: this.generation };
  }

  async initializePipelines(): Promise<void> {
    if (this.fillPipeline || this.destroyed) return;
    const fillModule = this.device.createShaderModule({
      label: "SVO fluid coverage fill",
      code: svoFluidCoverageFillShader(this.plan.cellsPerTexel),
    });
    this.fillLayout = this.device.createBindGroupLayout({
      label: "SVO fluid coverage fill layout",
      entries: [
        { binding: SVO_FLUID_COVERAGE_BINDINGS.coarsePhi, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: SVO_FLUID_COVERAGE_BINDINGS.params, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: SVO_FLUID_COVERAGE_BINDINGS.destination, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: SVO_FLUID_COVERAGE_LAYOUT.format, viewDimension: "3d" } },
      ],
    });
    this.fillPipeline = await this.device.createComputePipelineAsync({
      label: "Fill SVO fluid coverage",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fillLayout] }),
      compute: { module: fillModule, entryPoint: "fillFluidCoverage" },
    });
    const reduceModule = this.device.createShaderModule({
      label: "SVO fluid coverage reduce", code: svoFluidCoverageReduceShader,
    });
    this.reducePipeline = await this.device.createComputePipelineAsync({
      label: "Reduce SVO fluid coverage", layout: "auto",
      compute: { module: reduceModule, entryPoint: "reduceFluidCoverage" },
    });
  }

  frame(): ArrayBuffer {
    const live = this.visibleGeneration();
    return packSvoFluidCoverageFrame({
      origin_m: this.plan.origin_m,
      texelSize_m: this.plan.texelSize_m,
      dimensions: this.plan.dimensions,
      levelCount: this.plan.levelCount,
      valid: Boolean(live),
      generation: live?.generation ?? 0,
    });
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    this.allocate();
    if (!this.fillPipeline || !this.reducePipeline || !this.fillBindGroup || !this.texture) return;
    const fillPass = encoder.beginComputePass({ label: "Fill SVO fluid coverage volume" });
    fillPass.setPipeline(this.fillPipeline);
    fillPass.setBindGroup(0, this.fillBindGroup);
    const fill = dispatchCounts(this.plan.dimensions, SVO_FLUID_COVERAGE_LAYOUT.fillWorkgroupSize);
    fillPass.dispatchWorkgroups(fill[0], fill[1], fill[2]);
    fillPass.end();
    // One pass per level, not one pass with many dispatches. A compute pass is a
    // single usage scope, and each level's source subresource was the previous
    // level's storage-write destination: reading it as a sampled texture in the
    // same scope is a read/write conflict on that subresource. Separate passes
    // put the transition on a pass boundary, where the barrier is guaranteed.
    for (let level = 1; level < this.plan.levelCount; level += 1) {
      const pass = encoder.beginComputePass({ label: `Reduce SVO fluid coverage to level ${level}` });
      pass.setPipeline(this.reducePipeline);
      pass.setBindGroup(0, this.reduceBindGroups[level - 1]);
      const counts = dispatchCounts(levelDimensions(this.plan.dimensions, level), SVO_FLUID_COVERAGE_LAYOUT.reduceWorkgroupSize);
      pass.dispatchWorkgroups(counts[0], counts[1], counts[2]);
      pass.end();
    }
    this.generation += 1;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.texture?.destroy();
    this.uniforms?.destroy();
    this.texture = undefined;
    this.sampledView = undefined;
    this.fillBindGroup = undefined;
    this.reduceBindGroups = [];
    this.destroyed = true;
  }

  private allocate(): void {
    if (this.texture || !this.fillPipeline || !this.reducePipeline || !this.fillLayout) return;
    const { dimensions, levelCount } = this.plan;
    this.texture = this.device.createTexture({
      label: "SVO fluid coverage volume",
      size: [...dimensions],
      dimension: SVO_FLUID_COVERAGE_LAYOUT.dimension,
      format: SVO_FLUID_COVERAGE_LAYOUT.format,
      mipLevelCount: levelCount,
      // COPY_SRC is not needed to render; it exists so a readback can prove the
      // fill against the CPU coverage reference on a real device.
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.sampledView = this.texture.createView({ dimension: "3d" });

    const stride = Math.max(UNIFORM_ALIGNMENT, REDUCE_PARAM_WORDS * 4);
    this.uniforms = this.device.createBuffer({
      label: "SVO fluid coverage parameters",
      size: stride * Math.max(1, levelCount),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const fillParams = new ArrayBuffer(FILL_PARAM_WORDS * 4);
    const fillWords = new Uint32Array(fillParams), fillFloats = new Float32Array(fillParams);
    fillWords.set(this.plan.dimensions, 0);
    fillWords.set(this.plan.fieldDimensions, 4);
    fillFloats[8] = this.cellDiagonal_m;
    this.device.queue.writeBuffer(this.uniforms, 0, fillParams);

    this.fillBindGroup = this.device.createBindGroup({
      label: "SVO fluid coverage fill bindings",
      layout: this.fillLayout!,
      entries: [
        { binding: SVO_FLUID_COVERAGE_BINDINGS.coarsePhi, resource: this.sources.coarsePhi },
        { binding: SVO_FLUID_COVERAGE_BINDINGS.params, resource: { buffer: this.uniforms, offset: 0, size: FILL_PARAM_WORDS * 4 } },
        { binding: SVO_FLUID_COVERAGE_BINDINGS.destination, resource: this.texture.createView({ dimension: "3d", baseMipLevel: 0, mipLevelCount: 1 }) },
      ],
    });

    this.reduceBindGroups = [];
    for (let level = 1; level < levelCount; level += 1) {
      const destination = levelDimensions(dimensions, level), source = levelDimensions(dimensions, level - 1);
      const offset = stride * level;
      const params = new Uint32Array(REDUCE_PARAM_WORDS);
      params.set(destination, 0);
      params.set(source, 4);
      this.device.queue.writeBuffer(this.uniforms, offset, params);
      this.reduceBindGroups.push(this.device.createBindGroup({
        label: `SVO fluid coverage reduce level ${level}`,
        layout: this.reducePipeline.getBindGroupLayout(0),
        entries: [
          // Distinct, non-overlapping subresources: reading level-1 while writing
          // level is a legal single-pass usage of one texture.
          { binding: 0, resource: this.texture.createView({ dimension: "3d", baseMipLevel: level - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: this.texture.createView({ dimension: "3d", baseMipLevel: level, mipLevelCount: 1 }) },
          { binding: 2, resource: { buffer: this.uniforms, offset, size: REDUCE_PARAM_WORDS * 4 } },
        ],
      }));
    }
  }
}
