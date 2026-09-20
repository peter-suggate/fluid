import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { uniformVelocityExtrapolationShader } from "./webgpu-uniform-velocity-extrapolation.wgsl";

type Dims3 = readonly [number, number, number];

interface ExtrapolationPipelines {
  readonly clear: GPUComputePipeline;
  readonly seed: GPUComputePipeline;
  readonly update: GPUComputePipeline;
  readonly prepare: GPUComputePipeline;
  readonly resolve: GPUComputePipeline;
  readonly restrict: GPUComputePipeline;
  readonly prolong: GPUComputePipeline;
  readonly pack: GPUComputePipeline;
  readonly prolongPack: GPUComputePipeline;
  readonly coarseTable: GPUComputePipeline;
}

interface HierarchyLevel {
  readonly dims: Dims3;
  readonly down: GPUTexture;
  readonly up: GPUTexture;
  readonly originsDown?: GPUTexture;
  readonly originsUp?: GPUTexture;
}

interface FrontConfig {
  readonly sourceParity: number;
  readonly targetParity: number;
  readonly hierarchySourceUsesBaseDims?: boolean;
  readonly hierarchyTargetUsesBaseDims?: boolean;
  /** Pressure-active ABI level for this hierarchy target; -1 means direct. */
  readonly activeLevel?: number;
}

/**
 * The two halves of one Sec. 3.3 extension a trace seam can tell apart: the
 * narrow-band FIM front (seed → indirect sweeps → resolve) and the down/up
 * hierarchy fill that ends with the packed transport shell. The parent solver
 * maps these onto its own phase table; this module only names its seams.
 */
export type UniformExtrapolationTraceStage = "narrow-band-front" | "hierarchy-fill";

/**
 * Sec. 3.3 velocity extrapolation, with the geometric nearest-source fallback.
 *
 * This module deliberately owns its shader, narrow-band front state, hierarchy,
 * bind groups, and dispatch schedule. The parent solver supplies only the current
 * density/velocity fields and the already-published padded transport targets.
 */
export class WebGPUUniformVelocityExtrapolator {
  private readonly dummyOrigins: GPUTexture;
  private readonly dummyOriginsOut: GPUTexture;
  private readonly fusedGroups: GPUBindGroup[] = [];
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
  /**
   * The fill pyramid's own level dimensions, finest first.
   *
   * The shared active-region level table used to be seeded from the PRESSURE
   * hierarchy's level dimensions, with record = extension level + 1. That only
   * describes this pyramid while both hierarchies halve in lockstep, which
   * stops being true the moment the pressure plan semi-coarsens or is planned
   * on a window instead of the domain. Where that assumption no longer holds
   * the table is seeded from here instead.
   */
  get hierarchyLevelDimensions(): readonly (readonly [number, number, number])[] {
    return this.hierarchyLevels.map((level) => [...level.dims] as [number, number, number]);
  }
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
  /** The ceil(n/4) hierarchy level, when it exists and tiles the lattice exactly. */
  readonly coarseVelocityLevel?: GPUTexture;
  private readonly coarseTableGroup?: GPUBindGroup;
  private readonly activeFrontPassLimit: number;
  private activeFrontPasses: number;

  constructor(
    private readonly device: GPUDevice,
    private readonly dims: Dims3,
    _cellSize: Dims3,
    params: GPUBuffer,
    density: GPUTexture,
    faceOpen: GPUTexture,
    currentVelocity: GPUTexture,
    predictedVelocity: GPUTexture,
    currentTransport: GPUTexture,
    predictedTransport: GPUTexture,
    private readonly activeRegion: GPUBuffer,
    private readonly tileScratch: GPUBuffer,
    private readonly activeDispatch?: GPUBuffer,
    private readonly sourceAwareHierarchy = false,
    private readonly fuseTransportPack = sourceAwareHierarchy,
  ) {
    const [nx, ny, nz] = dims;
    const extent: Dims3 = [nx + 2, ny + 2, nz + 2];
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC;
    const scratch = (label: string) => device.createTexture({
      label, size: extent, dimension: "3d", format: "rgba32float", usage,
    });
    this.dummyOrigins = device.createTexture({ label: "Extension unused source origins", size: [1,1,1], dimension: "3d", format: "rgba32uint", usage });
    this.dummyOriginsOut = device.createTexture({ label: "Extension unused output origins", size: [1,1,1], dimension: "3d", format: "rgba32uint", usage });
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
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      // The parent's conditioning scratch, as one read_write binding: the 4h
      // table is written by one pass of its own and the class word is read by
      // the rest, so no dispatch ever sees it both ways.
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } },
    ] });
    this.pipelineLayout = device.createPipelineLayout({ label: "Uniform Sec. 3.3 extrapolation pipeline layout", bindGroupLayouts: [this.layout] });

    // The JRW07 solve is deliberately clipped to a two-cell accurate band;
    // farther air is owned by the hierarchy below. A domain-length ceiling
    // encoded hundreds of empty update/prepare pass pairs after this local
    // front had converged. Sixteen wavefronts cover the complete 26-neighbour
    // dependency diameter of that band, including cut-cell detours, while the
    // indirect active counter still proves termination rather than guessing it.
    this.activeFrontPassLimit = Math.min(Math.max(...dims), 16);
    this.activeFrontPasses = this.activeFrontPassLimit;

    const frontBuffer = (config: FrontConfig): GPUBuffer => {
      const buffer = device.createBuffer({
        label: `Uniform Sec. 3.3 front parity ${config.sourceParity}->${config.targetParity}`,
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, new Uint32Array([
        config.sourceParity,
        config.targetParity,
        config.hierarchySourceUsesBaseDims ? 1 : 0,
        config.hierarchyTargetUsesBaseDims ? 1 : 0,
        config.activeLevel === undefined || config.activeLevel < 0
          ? 0xffff_ffff : config.activeLevel,
        0, 0, 0,
      ]));
      this.levelBuffers.push(buffer);
      return buffer;
    };
    const dummyLevels = frontBuffer({ sourceParity: 0, targetParity: 0 });
    const group = (
      velocity: GPUTexture, primaryIn: GPUTexture, secondaryIn: GPUTexture,
      primaryOut: GPUTexture, secondaryOut: GPUTexture, levels: GPUBuffer = dummyLevels,
      preparesIndirect = false,
      originsIn = this.dummyOrigins, existingOrigins = this.dummyOrigins, originsOut = this.dummyOriginsOut,
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
      { binding: 11, resource: { buffer: this.activeRegion } },
      { binding: 12, resource: { buffer: this.tileScratch } },
      { binding: 13, resource: originsIn.createView() },
      { binding: 14, resource: existingOrigins.createView() },
      { binding: 15, resource: originsOut.createView() },
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
      { binding: 11, resource: { buffer: this.activeRegion } },
      { binding: 12, resource: { buffer: this.tileScratch } },
      { binding: 13, resource: this.dummyOrigins.createView() },
      { binding: 14, resource: this.dummyOrigins.createView() },
      { binding: 15, resource: this.dummyOriginsOut.createView() },
    ] });
    this.activeStateTexture = this.resolvedDistances;

    // CM11b Sec. 3.3.1 requires the reverse pass to leave every finest-grid
    // value known, which takes a coarsest level that sees liquid from every
    // cell: coarsen until EVERY axis is one cell, a collapsed axis staying at
    // one (ceil(1/2) = 1). Stopping when the shortest axis collapses -- the
    // tall-cell reading, whose coarsest level spans the short vertical --
    // covers that axis only. The paper's 2D figures run as a 128x128x8 slab,
    // which then topped out at 16x16x1: the fill reached one 8-cell block past
    // the liquid, every face beyond it packed as zero, and a falling ball's
    // lower surface met still air and was sliced flat at a block boundary
    // every step. Prolongation fills only unknown values, so levels above one
    // that is already fully known change nothing. Ceil division preserves
    // coverage for scene sizes that are not exact powers of two.
    let levelDims: Dims3 = dims;
    while (Math.max(...levelDims) > 1) {
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
      const origins = () => sourceAwareHierarchy ? device.createTexture({
        label: `Uniform nearest-source origins ${levelDims.join("x")}`,
        size: levelDims, dimension: "3d", format: "rgba32uint", usage,
      }) : undefined;
      this.hierarchyLevels.push({ dims: levelDims, down: levelTexture("down"), up: levelTexture("up"),
        originsDown: origins(), originsUp: origins() });
    }

    let finer = this.resolvedValues;
    let finerOrigins = this.dummyOrigins;
    for (let levelIndex = 0; levelIndex < this.hierarchyLevels.length; levelIndex += 1) {
      const level = this.hierarchyLevels[levelIndex];
      const hierarchyConfig = frontBuffer({
        sourceParity: 0, targetParity: 0,
        hierarchySourceUsesBaseDims: levelIndex === 0,
        activeLevel: this.activeDispatch && Math.min(...level.dims) > 1 ? levelIndex + 1 : -1,
      });
      this.hierarchyDownGroups.push(group(
        currentVelocity, finer, this.resolvedDistances, level.down, this.valuesB, hierarchyConfig, false, finerOrigins, this.dummyOrigins, level.originsDown,
      ));
      finer = level.down;
      finerOrigins = level.originsDown ?? this.dummyOrigins;
    }
    let coarser = this.hierarchyLevels.at(-1)?.down;
    let coarserOrigins = this.hierarchyLevels.at(-1)?.originsDown ?? this.dummyOrigins;
    for (let levelIndex = this.hierarchyLevels.length - 2; levelIndex >= -1 && coarser; levelIndex -= 1) {
      const existingFine = levelIndex >= 0
        ? this.hierarchyLevels[levelIndex].down
        : this.resolvedValues;
      const filledFine = levelIndex >= 0
        ? this.hierarchyLevels[levelIndex].up
        : this.valuesA;
      // As in the restrict: a level with a collapsed axis has no active-region
      // ABI entry and is dispatched whole. Only the continuation past the
      // shortest axis makes such a level a prolong target.
      const indirect = this.activeDispatch !== undefined
        && (levelIndex < 0 || Math.min(...this.hierarchyLevels[levelIndex].dims) > 1);
      const hierarchyConfig = frontBuffer({
        sourceParity: 0, targetParity: 0,
        hierarchyTargetUsesBaseDims: levelIndex < 0,
        activeLevel: indirect ? (levelIndex < 0 ? 0 : levelIndex + 1) : -1,
      });
      this.hierarchyUpGroups.push(group(
        currentVelocity, coarser, existingFine, filledFine, this.valuesB, hierarchyConfig, false,
        coarserOrigins, this.hierarchyLevels[levelIndex]?.originsDown, this.hierarchyLevels[levelIndex]?.originsUp,
      ));
      if (levelIndex < 0) {
        for (const target of [currentTransport, predictedTransport]) this.fusedGroups.push(group(
          currentVelocity, coarser, existingFine, target, this.valuesB, hierarchyConfig, false, coarserOrigins,
        ));
      }
      coarser = filledFine;
      coarserOrigins = this.hierarchyLevels[levelIndex]?.originsUp ?? this.dummyOrigins;
    }

    // If the grid is too small to have a hierarchy, the accurate narrow-band
    // result is already the complete paper-prescribed finest-level result.
    const packedValues = this.hierarchyLevels.length > 0 ? this.valuesA : this.resolvedValues;
    this.packCurrentGroup = group(currentVelocity, packedValues, this.valuesA, currentTransport, this.valuesB);
    this.packPredictedGroup = group(predictedVelocity, packedValues, this.valuesA, predictedTransport, this.valuesB);

    // The 4h level the parent's two-level sampler reads. Its prolong-filled
    // `up` texture is the complete field; when it is also the coarsest level
    // nothing prolongs into it, and its restricted `down` is the whole result.
    const coarse = this.hierarchyLevels[1];
    if (coarse && dims.every((value, axis) => value === 4 * coarse.dims[axis]!)) {
      this.coarseVelocityLevel = this.hierarchyLevels.length > 2 ? coarse.up : coarse.down;
      // The two storage-texture outputs are inert here but must still be
      // distinct subresources: one dispatch may not write the same texture
      // through two bindings.
      this.coarseTableGroup = group(
        currentVelocity, this.coarseVelocityLevel, this.resolvedDistances, this.valuesB, this.distancesB,
      );
    }
  }

  /** FIM scratch plus the explicit CM11b down/up velocity hierarchy. */
  get scratchBytes(): number {
    const [nx, ny, nz] = this.dims;
    const baseBytes = (nx + 2) * (ny + 2) * (nz + 2) * 6 * 16 + 40;
    const hierarchyBytes = this.hierarchyLevels.reduce(
      (sum, level) => sum + level.dims[0] * level.dims[1] * level.dims[2] * (this.sourceAwareHierarchy ? 4 : 2) * 16,
      0,
    );
    return baseBytes + hierarchyBytes + 32;
  }

  /** Hard wavefront ceiling; `frontPasses` is what an encode actually issues. */
  get activeFrontPassCeiling(): number { return this.activeFrontPassLimit; }

  get frontPasses(): number { return this.activeFrontPasses; }

  /** Live sweep budget. Below the band's dependency diameter the front is
   * resolved unconverged: updated faces keep their provisional value, and band
   * faces never reached stay unknown and fall to the hierarchy fill. */
  setFrontPasses(passes: number): void {
    this.activeFrontPasses = Number.isFinite(passes)
      ? Math.min(this.activeFrontPassLimit, Math.max(1, Math.round(passes))) : this.activeFrontPassLimit;
  }

  /** Four u32 words: active A/B counts, latest parity, and executed updates. */
  get convergenceDiagnostics(): GPUBuffer { return this.convergence; }

  get hierarchyLevelCount(): number { return this.hierarchyLevels.length; }

  /** Compute passes one `encode` call always encodes: the count is fixed at
   * construction; only the indirect front's workgroup counts are data-driven. */
  get encodedPassCount(): number {
    // seed + initial prepare + (update + prepare) per sweep + resolve + pack,
    // plus one restrict and one prolong per hierarchy traversal step.
    return 4 + 2 * this.activeFrontPasses
      + this.hierarchyDownGroups.length + this.hierarchyUpGroups.length
      - (this.fuseTransportPack && this.fusedGroups.length ? 1 : 0);
  }

  async initialize(signal?: AbortSignal): Promise<void> {
    if (this.pipelines) return;
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({
      label: "Uniform Sec. 3.3 extrapolation kernels",
      code: uniformVelocityExtrapolationShader,
    });
    const compile = (label: string, entryPoint: string) => compiler.compileComputePipeline({
      label, layout: this.pipelineLayout, compute: { module: shaderModule, entryPoint, constants: {
        SOURCE_AWARE_HIERARCHY: Number(this.sourceAwareHierarchy),
        ROOT_NX: this.dims[0], ROOT_NY: this.dims[1], ROOT_NZ: this.dims[2],
      } },
    }, { priority: "critical", signal });
    const [clear, seed, update, prepare, resolve, restrict, prolong, pack, coarseTable, prolongPack] = await Promise.all([
      compile("Uniform Sec. 3.3 clear sparse state", "clearExtrapolationState"),
      compile("Uniform Sec. 3.3 seed active front", "seedActiveFront"),
      compile("Uniform Sec. 3.3 update active front", "updateActiveFront"),
      compile("Uniform Sec. 3.3 prepare active dispatch", "prepareActiveDispatch"),
      compile("Uniform Sec. 3.3 resolve converged front", "resolveConvergedFront"),
      compile("Uniform Sec. 3.3 hierarchy restrict", "restrictKnownVelocity"),
      compile("Uniform Sec. 3.3 hierarchy prolong", "prolongUnknownVelocity"),
      compile("Uniform Sec. 3.3 transport shell", "packTransportShell"),
      compile("Uniform Sec. 3.3 publish 4h face table", "publishCoarseVelocityTable"),
      compile("Uniform nearest hierarchy and transport shell", "prolongAndPack"),
    ]);
    this.pipelines = { clear, seed, update, prepare, resolve, restrict, prolong, pack, coarseTable, prolongPack };
    const encoder = this.device.createCommandEncoder({ label: "Uniform Sec. 3.3 initialize sparse state" });
    for (const [label, group] of [["A", this.seedCurrentGroup], ["B", this.updateABGroup],
      ["resolved", this.resolveGroup]] as const) {
      const pass = encoder.beginComputePass({ label: `Uniform Sec. 3.3 clear ${label}` });
      pass.setPipeline(clear); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(this.dims[0] / 4), Math.ceil(this.dims[1] / 4), Math.ceil(this.dims[2] / 4));
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Group counts the host chose for this step, or undefined to keep taking
   * them from the GPU's indirect records. `base` sizes the finest passes (the
   * seed, resolve, prolong-to-base and pack, all of which read the main-grid
   * origin at word 7); `levels` is indexed by the shared active-region level
   * record, which hierarchy level i uses as record i+1. Only the launch size
   * changes: every origin still comes from `activeRegion`.
   */
  setWindowGroups(base?: readonly [number, number, number],
    levels?: readonly (readonly [number, number, number])[]): void {
    this.windowBaseGroups = base;
    this.windowLevelGroups = levels;
  }

  private windowBaseGroups?: readonly [number, number, number];
  private windowLevelGroups?: readonly (readonly [number, number, number])[];

  encode(
    encoder: GPUCommandEncoder,
    predicted: boolean,
    boundary?: (stage: UniformExtrapolationTraceStage) => void,
    publishCoarseTable = false,
  ): void {
    const pipelines = this.pipelines;
    if (!pipelines) throw new Error("Uniform Sec. 3.3 extrapolation pipelines are not initialized");
    encoder.clearBuffer(this.convergence);
    encoder.clearBuffer(this.dispatchArgs);
    const dispatchBase = (label: string, pipeline: GPUComputePipeline, group: GPUBindGroup) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      if (this.activeDispatch && this.windowBaseGroups) pass.dispatchWorkgroups(...this.windowBaseGroups);
      else if (this.activeDispatch) pass.dispatchWorkgroupsIndirect(this.activeDispatch, 13 * 4);
      else pass.dispatchWorkgroups(
        Math.ceil(this.dims[0] / 4), Math.ceil(this.dims[1] / 4), Math.ceil(this.dims[2] / 4));
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
      const restrictGroups = this.windowLevelGroups?.[levelIndex + 1];
      if (this.activeDispatch && Math.min(...level.dims) > 1 && restrictGroups) {
        pass.dispatchWorkgroups(...restrictGroups);
      } else if (this.activeDispatch && !this.windowBaseGroups && Math.min(...level.dims) > 1) {
        pass.dispatchWorkgroupsIndirect(this.activeDispatch, (16 + (levelIndex + 1) * 10 + 3) * 4);
      } else pass.dispatchWorkgroups(
        Math.ceil(level.dims[0] / 4), Math.ceil(level.dims[1] / 4), Math.ceil(level.dims[2] / 4));
      pass.end();
    }
    for (let passIndex = 0; passIndex < this.hierarchyUpGroups.length; passIndex += 1) {
      const levelIndex = this.hierarchyLevels.length - 2 - passIndex;
      const targetDims = levelIndex >= 0 ? this.hierarchyLevels[levelIndex].dims : this.dims;
      const pass = encoder.beginComputePass({ label: `${prefix} hierarchy prolong ${passIndex + 1}` });
      const fused = levelIndex < 0 && this.fuseTransportPack;
      pass.setPipeline(fused ? pipelines.prolongPack : pipelines.prolong);
      pass.setBindGroup(0, fused ? this.fusedGroups[predicted ? 1 : 0] : this.hierarchyUpGroups[passIndex]);
      const prolongGroups = levelIndex < 0 ? this.windowBaseGroups
        : this.windowLevelGroups?.[levelIndex + 1];
      if (this.activeDispatch && (levelIndex < 0 || Math.min(...targetDims) > 1) && prolongGroups) {
        pass.dispatchWorkgroups(...prolongGroups);
      } else if (this.activeDispatch && !this.windowBaseGroups
        && (levelIndex < 0 || Math.min(...targetDims) > 1)) {
        // A level past the shared level table has no record; host-sized mode
        // falls to the dense count below rather than a zero-size indirect one.
        pass.dispatchWorkgroupsIndirect(this.activeDispatch,
          levelIndex < 0 ? 13 * 4 : (16 + (levelIndex + 1) * 10 + 3) * 4);
      } else pass.dispatchWorkgroups(
        Math.ceil(targetDims[0] / 4), Math.ceil(targetDims[1] / 4), Math.ceil(targetDims[2] / 4));
      pass.end();
    }
    if (!this.fuseTransportPack || !this.fusedGroups.length) {
      const pass = encoder.beginComputePass({ label: `${prefix} pack transport shell` });
      pass.setPipeline(pipelines.pack);
      pass.setBindGroup(0, predicted ? this.packPredictedGroup : this.packCurrentGroup);
      if (this.activeDispatch && this.windowBaseGroups) pass.dispatchWorkgroups(...this.windowBaseGroups);
      else if (this.activeDispatch) pass.dispatchWorkgroupsIndirect(this.activeDispatch, 13 * 4);
      else pass.dispatchWorkgroups(
        Math.ceil((this.dims[0] + 2) / 4),
        Math.ceil((this.dims[1] + 2) / 4),
        Math.ceil((this.dims[2] + 2) / 4));
      pass.end();
    }
    if (publishCoarseTable) this.encodeCoarseVelocityTable(encoder);
    boundary?.("hierarchy-fill");
  }

  /**
   * One pass over ceil(n/4)^3 publishing the 4h level into the parent's face
   * table. It follows the hierarchy fill because it reads it; the caller asks
   * for it only while the two-level sampler is on.
   */
  encodeCoarseVelocityTable(encoder: GPUCommandEncoder): boolean {
    const level = this.hierarchyLevels[1];
    if (!this.pipelines || !this.coarseTableGroup || !level) return false;
    const pass = encoder.beginComputePass({ label: "Uniform Sec. 3.3 publish 4h face table" });
    pass.setPipeline(this.pipelines.coarseTable); pass.setBindGroup(0, this.coarseTableGroup);
    pass.dispatchWorkgroups(
      Math.ceil(level.dims[0] / 4), Math.ceil(level.dims[1] / 4), Math.ceil(level.dims[2] / 4));
    pass.end();
    return true;
  }

  /** True once the ceil(n/4) level exists and tiles the lattice exactly. */
  get coarseVelocityTableAvailable(): boolean { return this.coarseTableGroup !== undefined; }

  destroy(): void {
    this.dummyOrigins.destroy(); this.dummyOriginsOut.destroy();
    this.valuesA.destroy(); this.valuesB.destroy();
    this.distancesA.destroy(); this.distancesB.destroy();
    this.resolvedValues.destroy(); this.resolvedDistances.destroy();
    this.convergence.destroy();
    this.dispatchArgs.destroy();
    this.unusedDispatchStorage.destroy();
    this.hierarchyLevels.forEach((level) => { level.down.destroy(); level.up.destroy(); level.originsDown?.destroy(); level.originsUp?.destroy(); });
    this.levelBuffers.forEach((buffer) => buffer.destroy());
  }
}
