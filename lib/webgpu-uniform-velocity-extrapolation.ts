import { gpuCompilationManagerFor } from "./gpu-compilation-manager";
import { uniformVelocityExtrapolationShader } from "./webgpu-uniform-velocity-extrapolation.wgsl";

type Dims3 = readonly [number, number, number];

interface ExtrapolationPipelines {
  readonly seed: GPUComputePipeline;
  readonly update: GPUComputePipeline;
  readonly prepare: GPUComputePipeline;
  readonly resolve: GPUComputePipeline;
  readonly restrict: GPUComputePipeline;
  readonly prolong: GPUComputePipeline;
  readonly pack: GPUComputePipeline;
}

interface HierarchyLevel {
  readonly dims: Dims3;
  readonly down: GPUTexture;
  readonly up: GPUTexture;
}

interface FrontConfig {
  readonly sourceParity: number;
  readonly targetParity: number;
  readonly hierarchySourceUsesBaseDims?: boolean;
  readonly hierarchyTargetUsesBaseDims?: boolean;
}

/**
 * The two halves of one Sec. 3.3 extension a trace seam can tell apart: the
 * narrow-band FIM front (seed → indirect sweeps → resolve) and the down/up
 * hierarchy fill that ends with the packed transport shell. The parent solver
 * maps these onto its own phase table; this module only names its seams.
 */
export type UniformExtrapolationTraceStage = "narrow-band-front" | "hierarchy-fill";

/**
 * Paper-scoped Sec. 3.3 velocity extrapolation.
 *
 * This module deliberately owns its shader, narrow-band front state, hierarchy,
 * bind groups, and dispatch schedule. The parent solver supplies only the current
 * density/velocity fields and the already-published padded transport targets.
 */
export class WebGPUUniformVelocityExtrapolator {
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private pipelines?: ExtrapolationPipelines;
  private readonly valuesA: GPUTexture;
  private readonly valuesB: GPUTexture;
  private readonly distancesA: GPUTexture;
  private readonly distancesB: GPUTexture;
  private readonly resolvedValues: GPUTexture;
  private readonly resolvedDistances: GPUTexture;
  private readonly convergence: GPUBuffer;
  private readonly dispatchArgs: GPUBuffer;
  private readonly unusedDispatchStorage: GPUBuffer;
  private readonly levelBuffers: GPUBuffer[] = [];
  private readonly hierarchyLevels: HierarchyLevel[] = [];
  private readonly hierarchyDownGroups: GPUBindGroup[] = [];
  private readonly hierarchyUpGroups: GPUBindGroup[] = [];
  private readonly seedCurrentGroup: GPUBindGroup;
  private readonly seedPredictedGroup: GPUBindGroup;
  private readonly prepareSeedCurrentGroup: GPUBindGroup;
  private readonly prepareSeedPredictedGroup: GPUBindGroup;
  private readonly updateABGroup: GPUBindGroup;
  private readonly updateBAGroup: GPUBindGroup;
  private readonly prepareABGroup: GPUBindGroup;
  private readonly prepareBAGroup: GPUBindGroup;
  private readonly resolveGroup: GPUBindGroup;
  private readonly packCurrentGroup: GPUBindGroup;
  private readonly packPredictedGroup: GPUBindGroup;
  readonly activeStateTexture: GPUTexture;
  private readonly activeFrontPasses: number;

  constructor(
    private readonly device: GPUDevice,
    private readonly dims: Dims3,
    cellSize: Dims3,
    params: GPUBuffer,
    density: GPUTexture,
    faceOpen: GPUTexture,
    currentVelocity: GPUTexture,
    predictedVelocity: GPUTexture,
    currentTransport: GPUTexture,
    predictedTransport: GPUTexture,
  ) {
    const [nx, ny, nz] = dims;
    const extent: Dims3 = [nx + 2, ny + 2, nz + 2];
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC;
    const scratch = (label: string) => device.createTexture({
      label, size: extent, dimension: "3d", format: "rgba32float", usage,
    });
    this.valuesA = scratch("Uniform Sec. 3.3 FIM values A");
    this.valuesB = scratch("Uniform Sec. 3.3 FIM values B");
    this.distancesA = scratch("Uniform Sec. 3.3 FIM distances A");
    this.distancesB = scratch("Uniform Sec. 3.3 FIM distances B");
    this.resolvedValues = scratch("Uniform Sec. 3.3 resolved FIM values");
    this.resolvedDistances = scratch("Uniform Sec. 3.3 resolved FIM distances");
    this.convergence = device.createBuffer({
      label: "Uniform Sec. 3.3 active-front convergence",
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.dispatchArgs = device.createBuffer({
      label: "Uniform Sec. 3.3 indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.unusedDispatchStorage = device.createBuffer({
      label: "Uniform Sec. 3.3 unused dispatch storage binding",
      size: 12,
      usage: GPUBufferUsage.STORAGE,
    });

    this.layout = device.createBindGroupLayout({ label: "Uniform Sec. 3.3 extrapolation layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.pipelineLayout = device.createPipelineLayout({ label: "Uniform Sec. 3.3 extrapolation pipeline layout", bindGroupLayouts: [this.layout] });

    // CM11b Sec. 3.3 terminates only when the active list is empty.  Indirect
    // dispatch turns all remaining updates into zero-work calls once that
    // happens; a longest-axis ceiling gives every dependency chain enough
    // opportunities to settle without prematurely accepting a live front.
    this.activeFrontPasses = Math.max(...dims);

    const frontBuffer = (config: FrontConfig): GPUBuffer => {
      const buffer = device.createBuffer({
        label: `Uniform Sec. 3.3 front parity ${config.sourceParity}->${config.targetParity}`,
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, new Uint32Array([
        config.sourceParity,
        config.targetParity,
        config.hierarchySourceUsesBaseDims ? 1 : 0,
        config.hierarchyTargetUsesBaseDims ? 1 : 0,
      ]));
      this.levelBuffers.push(buffer);
      return buffer;
    };
    const dummyLevels = frontBuffer({ sourceParity: 0, targetParity: 0 });
    const group = (
      velocity: GPUTexture, primaryIn: GPUTexture, secondaryIn: GPUTexture,
      primaryOut: GPUTexture, secondaryOut: GPUTexture, levels: GPUBuffer = dummyLevels,
      preparesIndirect = false,
    ) => device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: velocity.createView() },
      { binding: 1, resource: density.createView() },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: primaryIn.createView() },
      { binding: 4, resource: secondaryIn.createView() },
      { binding: 5, resource: primaryOut.createView() },
      { binding: 6, resource: secondaryOut.createView() },
      { binding: 7, resource: { buffer: levels } },
      { binding: 8, resource: faceOpen.createView() },
      { binding: 9, resource: { buffer: this.convergence } },
      { binding: 10, resource: { buffer: preparesIndirect ? this.dispatchArgs : this.unusedDispatchStorage } },
    ] });

    this.seedCurrentGroup = group(currentVelocity, this.resolvedValues, this.resolvedDistances, this.valuesA, this.distancesA);
    this.seedPredictedGroup = group(predictedVelocity, this.resolvedValues, this.resolvedDistances, this.valuesA, this.distancesA);
    this.prepareSeedCurrentGroup = group(currentVelocity, this.resolvedValues, this.resolvedDistances, this.valuesA, this.distancesA, dummyLevels, true);
    this.prepareSeedPredictedGroup = group(predictedVelocity, this.resolvedValues, this.resolvedDistances, this.valuesA, this.distancesA, dummyLevels, true);
    const updateABLevels = frontBuffer({ sourceParity: 0, targetParity: 1 });
    const updateBALevels = frontBuffer({ sourceParity: 1, targetParity: 0 });
    this.updateABGroup = group(currentVelocity, this.valuesA, this.distancesA, this.valuesB, this.distancesB, updateABLevels);
    this.updateBAGroup = group(currentVelocity, this.valuesB, this.distancesB, this.valuesA, this.distancesA, updateBALevels);
    this.prepareABGroup = group(currentVelocity, this.valuesA, this.distancesA, this.valuesB, this.distancesB, updateABLevels, true);
    this.prepareBAGroup = group(currentVelocity, this.valuesB, this.distancesB, this.valuesA, this.distancesA, updateBALevels, true);
    this.resolveGroup = device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: this.valuesB.createView() },
      { binding: 1, resource: this.distancesB.createView() },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: this.valuesA.createView() },
      { binding: 4, resource: this.distancesA.createView() },
      { binding: 5, resource: this.resolvedValues.createView() },
      { binding: 6, resource: this.resolvedDistances.createView() },
      { binding: 7, resource: { buffer: dummyLevels } },
      { binding: 8, resource: faceOpen.createView() },
      { binding: 9, resource: { buffer: this.convergence } },
      { binding: 10, resource: { buffer: this.dispatchArgs } },
    ] });
    this.activeStateTexture = this.resolvedDistances;

    // CM11b Sec. 3.3.1 requires the reverse pass to leave every finest-grid
    // value known. Its coarsest tall-cell level spans the short axis. On this
    // regular-grid adaptation, include the /2 level that likewise collapses
    // the shortest extent to one cell; stopping one level earlier leaves an
    // unsupported upper slab. Ceil division preserves coverage for scene
    // sizes that are not exact powers of two.
    let levelDims: Dims3 = dims;
    while (Math.min(...levelDims) > 1) {
      levelDims = [
        Math.ceil(levelDims[0] / 2),
        Math.ceil(levelDims[1] / 2),
        Math.ceil(levelDims[2] / 2),
      ];
      const levelTexture = (direction: "down" | "up") => device.createTexture({
        label: `Uniform Sec. 3.3 hierarchy ${direction} ${levelDims.join("x")}`,
        size: levelDims,
        dimension: "3d",
        format: "rgba32float",
        usage,
      });
      this.hierarchyLevels.push({ dims: levelDims, down: levelTexture("down"), up: levelTexture("up") });
    }

    let finer = this.resolvedValues;
    for (let levelIndex = 0; levelIndex < this.hierarchyLevels.length; levelIndex += 1) {
      const level = this.hierarchyLevels[levelIndex];
      const hierarchyConfig = frontBuffer({
        sourceParity: 0, targetParity: 0,
        hierarchySourceUsesBaseDims: levelIndex === 0,
      });
      this.hierarchyDownGroups.push(group(
        currentVelocity, finer, this.resolvedDistances, level.down, this.valuesB, hierarchyConfig,
      ));
      finer = level.down;
    }
    let coarser = this.hierarchyLevels.at(-1)?.down;
    for (let levelIndex = this.hierarchyLevels.length - 2; levelIndex >= -1 && coarser; levelIndex -= 1) {
      const existingFine = levelIndex >= 0
        ? this.hierarchyLevels[levelIndex].down
        : this.resolvedValues;
      const filledFine = levelIndex >= 0
        ? this.hierarchyLevels[levelIndex].up
        : this.valuesA;
      const hierarchyConfig = frontBuffer({
        sourceParity: 0, targetParity: 0,
        hierarchyTargetUsesBaseDims: levelIndex < 0,
      });
      this.hierarchyUpGroups.push(group(
        currentVelocity, coarser, existingFine, filledFine, this.valuesB, hierarchyConfig,
      ));
      coarser = filledFine;
    }

    // If the grid is too small to have a hierarchy, the accurate narrow-band
    // result is already the complete paper-prescribed finest-level result.
    const packedValues = this.hierarchyLevels.length > 0 ? this.valuesA : this.resolvedValues;
    this.packCurrentGroup = group(currentVelocity, packedValues, this.valuesA, currentTransport, this.valuesB);
    this.packPredictedGroup = group(predictedVelocity, packedValues, this.valuesA, predictedTransport, this.valuesB);
  }

  /** FIM scratch plus the explicit CM11b down/up velocity hierarchy. */
  get scratchBytes(): number {
    const [nx, ny, nz] = this.dims;
    const baseBytes = (nx + 2) * (ny + 2) * (nz + 2) * 6 * 16 + 40;
    const hierarchyBytes = this.hierarchyLevels.reduce(
      (sum, level) => sum + level.dims[0] * level.dims[1] * level.dims[2] * 2 * 16,
      0,
    );
    return baseBytes + hierarchyBytes;
  }

  get activeFrontPassCeiling(): number { return this.activeFrontPasses; }

  /** Four u32 words: active A/B counts, latest parity, and executed updates. */
  get convergenceDiagnostics(): GPUBuffer { return this.convergence; }

  get hierarchyLevelCount(): number { return this.hierarchyLevels.length; }

  /** Compute passes one `encode` call always encodes: the count is fixed at
   * construction; only the indirect front's workgroup counts are data-driven. */
  get encodedPassCount(): number {
    // seed + initial prepare + (update + prepare) per sweep + resolve + pack,
    // plus one restrict and one prolong per hierarchy traversal step.
    return 4 + 2 * this.activeFrontPasses
      + this.hierarchyDownGroups.length + this.hierarchyUpGroups.length;
  }

  async initialize(signal?: AbortSignal): Promise<void> {
    if (this.pipelines) return;
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({
      label: "Uniform Sec. 3.3 extrapolation kernels",
      code: uniformVelocityExtrapolationShader,
    });
    const compile = (label: string, entryPoint: string) => compiler.compileComputePipeline({
      label, layout: this.pipelineLayout, compute: { module: shaderModule, entryPoint },
    }, { priority: "critical", signal });
    const [seed, update, prepare, resolve, restrict, prolong, pack] = await Promise.all([
      compile("Uniform Sec. 3.3 seed active front", "seedActiveFront"),
      compile("Uniform Sec. 3.3 update active front", "updateActiveFront"),
      compile("Uniform Sec. 3.3 prepare active dispatch", "prepareActiveDispatch"),
      compile("Uniform Sec. 3.3 resolve converged front", "resolveConvergedFront"),
      compile("Uniform Sec. 3.3 hierarchy restrict", "restrictKnownVelocity"),
      compile("Uniform Sec. 3.3 hierarchy prolong", "prolongUnknownVelocity"),
      compile("Uniform Sec. 3.3 transport shell", "packTransportShell"),
    ]);
    this.pipelines = { seed, update, prepare, resolve, restrict, prolong, pack };
  }

  encode(
    encoder: GPUCommandEncoder,
    predicted: boolean,
    boundary?: (stage: UniformExtrapolationTraceStage) => void,
  ): void {
    const pipelines = this.pipelines;
    if (!pipelines) throw new Error("Uniform Sec. 3.3 extrapolation pipelines are not initialized");
    encoder.clearBuffer(this.convergence);
    encoder.clearBuffer(this.dispatchArgs);
    const dispatchBase = (label: string, pipeline: GPUComputePipeline, group: GPUBindGroup) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(
        Math.ceil(this.dims[0] / 4), Math.ceil(this.dims[1] / 4), Math.ceil(this.dims[2] / 4),
      );
      pass.end();
    };
    const prefix = predicted ? "Uniform predicted Sec. 3.3" : "Uniform Sec. 3.3";
    dispatchBase(`${prefix} seed active front`, pipelines.seed,
      predicted ? this.seedPredictedGroup : this.seedCurrentGroup);
    const prepare = (label: string, group: GPUBindGroup) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipelines.prepare); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    };
    prepare(`${prefix} prepare initial active dispatch`,
      predicted ? this.prepareSeedPredictedGroup : this.prepareSeedCurrentGroup);
    for (let iteration = 0; iteration < this.activeFrontPasses; iteration += 1) {
      const group = iteration % 2 === 0 ? this.updateABGroup : this.updateBAGroup;
      const pass = encoder.beginComputePass({ label: `${prefix} FIM indirect update ${iteration + 1}` });
      pass.setPipeline(pipelines.update); pass.setBindGroup(0, group);
      pass.dispatchWorkgroupsIndirect(this.dispatchArgs, 0); pass.end();
      prepare(`${prefix} prepare active dispatch ${iteration + 2}`,
        iteration % 2 === 0 ? this.prepareABGroup : this.prepareBAGroup);
    }
    dispatchBase(`${prefix} resolve converged front`, pipelines.resolve, this.resolveGroup);
    boundary?.("narrow-band-front");
    for (let levelIndex = 0; levelIndex < this.hierarchyDownGroups.length; levelIndex += 1) {
      const level = this.hierarchyLevels[levelIndex];
      const pass = encoder.beginComputePass({ label: `${prefix} hierarchy restrict ${levelIndex + 1}` });
      pass.setPipeline(pipelines.restrict); pass.setBindGroup(0, this.hierarchyDownGroups[levelIndex]);
      pass.dispatchWorkgroups(
        Math.ceil(level.dims[0] / 4), Math.ceil(level.dims[1] / 4), Math.ceil(level.dims[2] / 4),
      );
      pass.end();
    }
    for (let passIndex = 0; passIndex < this.hierarchyUpGroups.length; passIndex += 1) {
      const levelIndex = this.hierarchyLevels.length - 2 - passIndex;
      const targetDims = levelIndex >= 0 ? this.hierarchyLevels[levelIndex].dims : this.dims;
      const pass = encoder.beginComputePass({ label: `${prefix} hierarchy prolong ${passIndex + 1}` });
      pass.setPipeline(pipelines.prolong); pass.setBindGroup(0, this.hierarchyUpGroups[passIndex]);
      pass.dispatchWorkgroups(
        Math.ceil(targetDims[0] / 4), Math.ceil(targetDims[1] / 4), Math.ceil(targetDims[2] / 4),
      );
      pass.end();
    }
    const pass = encoder.beginComputePass({ label: `${prefix} pack transport shell` });
    pass.setPipeline(pipelines.pack);
    pass.setBindGroup(0, predicted ? this.packPredictedGroup : this.packCurrentGroup);
    pass.dispatchWorkgroups(
      Math.ceil((this.dims[0] + 2) / 4),
      Math.ceil((this.dims[1] + 2) / 4),
      Math.ceil((this.dims[2] + 2) / 4),
    );
    pass.end();
    boundary?.("hierarchy-fill");
  }

  destroy(): void {
    this.valuesA.destroy(); this.valuesB.destroy();
    this.distancesA.destroy(); this.distancesB.destroy();
    this.resolvedValues.destroy(); this.resolvedDistances.destroy();
    this.convergence.destroy();
    this.dispatchArgs.destroy();
    this.unusedDispatchStorage.destroy();
    this.hierarchyLevels.forEach((level) => { level.down.destroy(); level.up.destroy(); });
    this.levelBuffers.forEach((buffer) => buffer.destroy());
  }
}
