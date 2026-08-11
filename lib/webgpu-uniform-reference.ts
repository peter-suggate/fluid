import {
  combineInitialBrickWet,
  damBreakBoxContains,
  initialFluidBrickContainsCell,
  sceneDamBreakBox,
} from "./initial-fluid";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "./inflow-boundary";
import type { SceneDescription } from "./model";
import { planUniformHostAllocation } from "./uniform-host-allocation";
import { initializeRigidBodies, type RigidBodyState } from "./rigid-body";
import { sceneLatticeDimensions } from "./scene-lattice";
import { planGPUAdvance } from "./tall-cell-diagnostics";
import type { GPUQuality } from "./tall-cell-grid";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  type GPUEulerianInfo,
  type GPURigidLoad,
} from "./webgpu-eulerian";
import { GPUInitializationTaskRunner, type GPUInitializationTask } from "./gpu-initialization";
import type { GPUSolverInstance, GPUInitializationReporter } from "./methods/types";
import { WebGPURigidBodySystem } from "./webgpu-rigid-body";
import { uniformReferenceComputeShader } from "./webgpu-uniform-reference.wgsl";

export interface WebGPUUniformReferenceOptions {
  pressureIterations?: number;
  deferPipelineCompilation?: boolean;
}

interface UniformReferencePipelines {
  advect: GPUComputePipeline;
  reverse: GPUComputePipeline;
  correct: GPUComputePipeline;
  jacobi: GPUComputePipeline;
  project: GPUComputePipeline;
  coupleRigid: GPUComputePipeline;
  reduce: GPUComputePipeline;
  occupancy: GPUComputePipeline;
  transport: GPUComputePipeline;
  fluxScales: GPUComputePipeline;
  measureVolume: GPUComputePipeline;
  correctVolume: GPUComputePipeline;
  smoothSurface: GPUComputePipeline;
  sharpenCompute: GPUComputePipeline;
  sharpenScatter: GPUComputePipeline;
  sharpenResolve: GPUComputePipeline;
}

const PIPELINES = [
  ["advect", "Advect velocity", "advect", false],
  ["reverse", "Reverse advection", "reverseAdvection", false],
  ["correct", "Correct advection", "correctAdvection", false],
  ["jacobi", "Relax pressure", "jacobi", false],
  ["project", "Project velocity", "project", false],
  ["coupleRigid", "Couple rigid bodies", "coupleRigid", false],
  ["reduce", "Reduce diagnostics", "reduceDiagnostics", false],
  ["occupancy", "Build occupancy", "buildOccupancy", false],
  ["transport", "Build transport field", "buildTransport", true],
  ["fluxScales", "Build flux scales", "buildFluxScales", true],
  ["measureVolume", "Measure conservative volume", "measureVolumeCorrection", false],
  ["correctVolume", "Correct conservative volume", "applyVolumeCorrection", false],
  ["smoothSurface", "Smooth presentation surface", "smoothSurface", false],
  ["sharpenCompute", "Compute interface sharpening", "sharpenCompute", false],
  ["sharpenScatter", "Scatter conserved interface mass", "sharpenScatter", false],
  ["sharpenResolve", "Resolve conserved interface mass", "sharpenResolve", false],
] as const;

const pipelineCache = new WeakMap<GPUDevice, UniformReferencePipelines>();

/**
 * Standalone dense reference implementation.
 *
 * This class deliberately has no octree option or adaptive compatibility
 * branch. Its allocations, pipelines, step graph, and diagnostics are owned
 * entirely by the `uniform` method plugin.
 */
export class WebGPUUniformReferenceSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly columnBaseTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;

  private readonly velocityA: GPUTexture;
  private readonly velocityB: GPUTexture;
  private readonly velocityC: GPUTexture;
  private readonly velocityD: GPUTexture;
  private readonly pressureA: GPUTexture;
  private readonly pressureB: GPUTexture;
  private readonly volumeA: GPUTexture;
  private readonly volumeB: GPUTexture;
  private readonly surfaceA: GPUTexture;
  private readonly surfaceB: GPUTexture;
  private readonly heightA: GPUTexture;
  private readonly heightB: GPUTexture;
  private readonly terrainTexture: GPUTexture;
  private readonly transportA: GPUTexture;
  private readonly transportB: GPUTexture;
  private readonly fluxScales: GPUTexture;
  private readonly params: GPUBuffer;
  private readonly reductions: GPUBuffer;
  private readonly conditioningScratch: GPUBuffer;
  private readonly rigidExchange: GPUBuffer;
  private readonly rigidSystem: WebGPURigidBodySystem;
  private readonly mainLayout: GPUBindGroupLayout;
  private readonly prepLayout: GPUBindGroupLayout;
  private readonly mainPipelineLayout: GPUPipelineLayout;
  private readonly prepPipelineLayout: GPUPipelineLayout;
  private readonly occupancyGroup: GPUBindGroup;
  private readonly advectGroup: GPUBindGroup;
  private readonly reverseGroup: GPUBindGroup;
  private readonly correctGroup: GPUBindGroup;
  private readonly jacobiABGroup: GPUBindGroup;
  private readonly jacobiBAGroup: GPUBindGroup;
  private readonly projectGroup: GPUBindGroup;
  private readonly rigidGroup: GPUBindGroup;
  private readonly reductionGroup: GPUBindGroup;
  private readonly volumeMeasureGroup: GPUBindGroup;
  private readonly volumeCorrectionGroup: GPUBindGroup;
  private readonly smoothSurfaceAGroup: GPUBindGroup;
  private readonly smoothSurfaceBGroup: GPUBindGroup;
  private readonly sharpenComputeGroup: GPUBindGroup;
  private readonly sharpenScatterGroup: GPUBindGroup;
  private readonly sharpenResolveGroup: GPUBindGroup;
  private readonly transportCurrentGroup: GPUBindGroup;
  private readonly transportPredictedGroup: GPUBindGroup;
  private readonly fluxScaleGroup: GPUBindGroup;
  private shaderModule?: GPUShaderModule;
  private pipelines?: UniformReferencePipelines;
  private inflowBoundary?: InflowGridBoundary;
  private statsReadback?: GPUBuffer;
  private readbackPending = false;
  private lastTime = 0;
  private referenceVolumeCells = 0;
  private disposed = false;

  constructor(
    private readonly device: GPUDevice,
    public scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformReferenceOptions = {},
  ) {
    const [nx, ny, nz] = sceneLatticeDimensions(scene, device.limits.maxTextureDimension3D);
    const allocation = planUniformHostAllocation(nx, ny, nz, "maccormack");
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture3d = (label: string, format: GPUTextureFormat, size: GPUExtent3D) =>
      device.createTexture({ label, size, dimension: "3d", format, usage });
    const velocity = (label: string) => texture3d(label, "rgba32float", allocation.velocityExtent);
    const scalar = (label: string) => texture3d(label, "r32float", allocation.volumeExtent);
    this.velocityA = velocity("Uniform reference velocity A");
    this.velocityB = velocity("Uniform reference velocity B");
    this.velocityC = velocity("Uniform reference velocity C");
    this.velocityD = velocity("Uniform reference velocity D");
    this.pressureA = scalar("Uniform reference pressure A");
    this.pressureB = scalar("Uniform reference pressure B");
    this.volumeA = scalar("Uniform reference volume A");
    this.volumeB = scalar("Uniform reference volume B");
    this.surfaceA = scalar("Uniform reference smoothed surface A");
    this.surfaceB = scalar("Uniform reference smoothed surface B");
    this.heightA = device.createTexture({ label: "Uniform reference column base", size: [nx, nz], format: "rg32float", usage });
    this.heightB = device.createTexture({ label: "Uniform reference column occupancy", size: [nx, nz], format: "rg32float", usage });
    this.terrainTexture = device.createTexture({ label: "Uniform reference terrain", size: [nx, nz], format: "r32float", usage });
    const transportUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    this.transportA = device.createTexture({ label: "Uniform reference transport A", size: allocation.transportExtent, dimension: "3d", format: "rgba16float", usage: transportUsage });
    this.transportB = device.createTexture({ label: "Uniform reference transport B", size: allocation.transportExtent, dimension: "3d", format: "rgba16float", usage: transportUsage });
    this.fluxScales = device.createTexture({ label: "Uniform reference flux scales", size: allocation.fluxExtent, dimension: "3d", format: "rg32float", usage: transportUsage });
    this.params = device.createBuffer({ label: "Uniform reference parameters", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductions = device.createBuffer({ label: "Uniform reference diagnostics and volume control", size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.conditioningScratch = device.createBuffer({ label: "Uniform reference compatibility scratch", size: allocation.conditioningBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.rigidExchange = device.createBuffer({ label: "Uniform reference rigid exchange", size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidSystem = new WebGPURigidBodySystem(device, scene, this.rigidExchange,
      this.terrainTexture, options.deferPipelineCompilation);
    this.rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [nx, ny, nz]) : undefined;

    this.mainLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 17, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 21, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
    ] });
    this.prepLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
      { binding: 18, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "3d" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    ] });
    this.mainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.mainLayout] });
    this.prepPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.prepLayout] });
    const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    const group = (velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture,
      pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture,
      heightIn: GPUTexture, heightOut: GPUTexture, predicted = velocityIn,
      reversed = velocityIn, transport = this.transportA, surface = volumeIn) => device.createBindGroup({
        layout: this.mainLayout, entries: [
          { binding: 0, resource: velocityIn.createView() }, { binding: 1, resource: velocityOut.createView() },
          { binding: 2, resource: pressureIn.createView() }, { binding: 3, resource: pressureOut.createView() },
          { binding: 4, resource: volumeIn.createView() }, { binding: 5, resource: volumeOut.createView() },
          { binding: 6, resource: { buffer: this.params } }, { binding: 7, resource: heightIn.createView() },
          { binding: 8, resource: heightOut.createView() }, { binding: 9, resource: { buffer: this.reductions } },
          { binding: 10, resource: { buffer: this.rigidSystem.stateBuffer } }, { binding: 11, resource: { buffer: this.rigidExchange } },
          { binding: 12, resource: predicted.createView() }, { binding: 13, resource: reversed.createView() },
          { binding: 14, resource: transport.createView() }, { binding: 15, resource: sampler },
          { binding: 17, resource: this.fluxScales.createView() }, { binding: 19, resource: { buffer: this.conditioningScratch } },
          { binding: 20, resource: surface.createView() }, { binding: 21, resource: this.terrainTexture.createView() },
        ],
      });
    const prep = (velocityIn: GPUTexture, transportOut: GPUTexture) => device.createBindGroup({
      layout: this.prepLayout, entries: [
        { binding: 0, resource: velocityIn.createView() }, { binding: 4, resource: this.volumeA.createView() },
        { binding: 6, resource: { buffer: this.params } }, { binding: 16, resource: transportOut.createView() },
        { binding: 18, resource: this.fluxScales.createView() }, { binding: 20, resource: this.volumeA.createView() },
      ],
    });
    this.occupancyGroup = group(this.velocityA, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB);
    this.advectGroup = group(this.velocityA, this.velocityC, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityB, this.velocityD, this.transportA);
    this.reverseGroup = group(this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityB, this.transportB);
    this.correctGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityC, this.velocityD, this.transportA, this.volumeB);
    this.jacobiABGroup = group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.jacobiBAGroup = group(this.velocityB, this.velocityA, this.pressureB, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA);
    const pressureIterations = Math.max(16, Math.min(400, Math.round(options.pressureIterations ?? 64)));
    const finalPressure = pressureIterations % 2 === 0 ? this.pressureA : this.pressureB;
    const sparePressure = pressureIterations % 2 === 0 ? this.pressureB : this.pressureA;
    this.projectGroup = group(this.velocityB, this.velocityA, finalPressure, sparePressure, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.rigidGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.reductionGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.volumeMeasureGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.volumeCorrectionGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.smoothSurfaceAGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.surfaceA, this.heightB, this.heightA);
    this.smoothSurfaceBGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.surfaceA, this.surfaceB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeA);
    this.sharpenComputeGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenScatterGroup = group(this.velocityA, this.velocityB, this.pressureB, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.sharpenResolveGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.transportCurrentGroup = prep(this.velocityA, this.transportA);
    this.transportPredictedGroup = prep(this.velocityC, this.transportB);
    this.fluxScaleGroup = prep(this.velocityA, this.transportA);

    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count,
      regularLayers: ny, maximumNeighborDelta: 0, gridKind: "uniform",
      cellSize_m: Math.min(scene.container.width_m / nx, scene.container.height_m / ny, scene.container.depth_m / nz),
      pressureIterations, pressureSolver: "Uniform weighted Jacobi",
      allocatedBytes: allocation.allocatedBytes, quality,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0,
      simulationLag_s: 0, encodedSteps: 0, maximumTallCellHeight: 0,
      hostFluidAuthority: "gpu-resident", hostSimulationSizedWorkItems: 0,
      hostSchedulingUsesReadback: false,
      volumeControl: true, volumeCorrectionNormalSpeed_cells_s: 2,
    };
    this.volumeTexture = this.volumeA;
    this.surfaceFieldTexture = this.surfaceB;
    this.columnBaseTexture = this.heightA;
    this.velocityTexture = this.velocityA;
    this.initializeVolumeAndTerrain();
    if (!options.deferPipelineCompilation) {
      this.compilePipelinesSync();
      this.encodeInitialPresentationSurface();
    }
  }

  static async createAsync(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: WebGPUUniformReferenceOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUUniformReferenceSolver> {
    const runner = new GPUInitializationTaskRunner(onProgress, signal);
    let solver: WebGPUUniformReferenceSolver | undefined;
    try {
      await runner.run([{ id: "uniform.allocate", phase: "allocation", label: "Allocate uniform reference resources", run: () => {
        solver = new WebGPUUniformReferenceSolver(device, scene, quality, onRigidLoads, { ...options, deferPipelineCompilation: true });
      } }]);
      await runner.run(solver!.initializationTasks());
      return solver!;
    } catch (error) {
      solver?.destroy();
      throw error;
    }
  }

  private pipelineDescriptor(entryPoint: string, prep: boolean): GPUComputePipelineDescriptor {
    this.shaderModule ??= this.device.createShaderModule({ label: "Uniform reference kernels", code: uniformReferenceComputeShader });
    return { layout: prep ? this.prepPipelineLayout : this.mainPipelineLayout, compute: { module: this.shaderModule, entryPoint } };
  }

  private compilePipelinesSync(): void {
    const compiled = Object.fromEntries(PIPELINES.map(([key, , entryPoint, prep]) => [key,
      this.device.createComputePipeline(this.pipelineDescriptor(entryPoint, prep))])) as unknown as UniformReferencePipelines;
    this.pipelines = compiled;
    pipelineCache.set(this.device, compiled);
  }

  private initializationTasks(): GPUInitializationTask[] {
    const cached = pipelineCache.get(this.device);
    const tasks = [...this.rigidSystem.initializationTasks()];
    let pipelineReadyId: string;
    if (cached) {
      pipelineReadyId = "uniform.pipeline.cache";
      tasks.push({ id: pipelineReadyId, phase: "solver-pipelines", label: "Reuse uniform reference programs", run: () => { this.pipelines = cached; } });
    } else {
      const compiled: Partial<UniformReferencePipelines> = {};
      const ids = PIPELINES.map(([key]) => `uniform.pipeline.${key}`);
      PIPELINES.forEach(([key, label, entryPoint, prep], index) => tasks.push({
        id: ids[index], phase: "solver-pipelines", label,
        run: async () => { compiled[key] = await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint, prep)); },
      }));
      pipelineReadyId = "uniform.pipeline.publish";
      tasks.push({ id: pipelineReadyId, phase: "solver-pipelines", label: "Publish uniform reference programs", dependencies: ids, run: () => {
        this.pipelines = compiled as UniformReferencePipelines;
        pipelineCache.set(this.device, this.pipelines);
      } });
    }
    tasks.push({ id: "uniform.surface.initial", phase: "upload", label: "Reconstruct smooth t=0 surface", dependencies: [pipelineReadyId], run: () => { this.encodeInitialPresentationSurface(); } });
    tasks.push({ id: "uniform.warmup", phase: "warmup", label: "Fence uniform t=0 uploads", dependencies: ["uniform.surface.initial"], run: async () => { await this.device.queue.onSubmittedWorkDone(); } });
    return tasks;
  }

  private encodeInitialPresentationSurface(): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference t=0 surface reconstruction" });
    this.run(encoder, "Uniform initial presentation smoothing A", this.pipelines.smoothSurface, this.smoothSurfaceAGroup);
    this.run(encoder, "Uniform initial presentation smoothing B", this.pipelines.smoothSurface, this.smoothSurfaceBGroup);
    this.device.queue.submit([encoder.finish()]);
  }

  private initializeVolumeAndTerrain(): void {
    const { nx, ny, nz } = this.info;
    const c = this.scene.container;
    const terrain = terrainColumnHeights(this.scene, nx, nz);
    const cellHeight = c.height_m / ny;
    const volume = new Float32Array(nx * ny * nz);
    const dam = sceneDamBreakBox(this.scene);
    let initial = 0;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const aboveGround = (y + 0.5) * cellHeight > terrain[x + nx * z];
      const brick = initialFluidBrickContainsCell(this.scene, x, y, z, [nx, ny, nz]);
      const base = this.scene.fluid.initialCondition === "dam-break"
        ? damBreakBoxContains(dam, (x + 0.5) / nx, (y + 0.5) / ny, (z + 0.5) / nz)
        : (y + 0.5) / ny <= c.fillFraction;
      const wet = aboveGround && combineInitialBrickWet(this.scene, brick, base);
      volume[x + nx * (y + ny * z)] = wet ? 1 : 0;
      if (wet) initial += 1;
    }
    this.upload3DF32(this.volumeA, volume, nx, ny, nz);
    this.upload3DF32(this.volumeB, volume, nx, ny, nz);
    this.upload3DF32(this.surfaceA, volume, nx, ny, nz);
    this.upload3DF32(this.surfaceB, volume, nx, ny, nz);
    this.referenceVolumeCells = initial;
    const terrainCells = Float32Array.from(terrain, (height) => height / cellHeight);
    this.upload2DF32(this.terrainTexture, terrainCells, nx, nz);
    Object.assign(this.info, {
      initialVolumeCellSum: initial, volumeCellSum: initial,
      representedVolumeCellSum: initial, volumeDrift: 0, representedVolumeDrift: 0,
      rawVolumeDrift: 0, volumeTelemetrySource: "initial-condition",
      maxSpeed_m_s: 0,
      front_m: this.scene.fluid.initialCondition === "dam-break"
        ? -c.width_m / 2 + dam.max.x * c.width_m : c.width_m / 2,
      frontTelemetrySource: "initial-condition",
    });
  }

  private upload3DF32(texture: GPUTexture, values: Float32Array, nx: number, ny: number, nz: number): void {
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * ny * nz), source = new Uint8Array(values.buffer);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      const row = y + ny * z;
      packed.set(source.subarray(rowBytes * row, rowBytes * (row + 1)), padded * row);
    }
    this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
  }

  private upload2DF32(texture: GPUTexture, values: Float32Array, nx: number, nz: number): void {
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * nz), source = new Uint8Array(values.buffer);
    for (let z = 0; z < nz; z += 1) packed.set(source.subarray(rowBytes * z, rowBytes * (z + 1)), padded * z);
    this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: nz }, { width: nx, height: nz });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup): void {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
  }

  private run(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, group: GPUBindGroup): void {
    const pass = encoder.beginComputePass({ label });
    this.dispatch(pass, pipeline, group);
    pass.end();
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []): boolean {
    if (this.disposed) return false;
    const advance = planGPUAdvance(time_s, this.lastTime, this.scene.numerics.maxDt_s);
    if (!advance) return false;
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    const dt = advance.dt_s;
    this.lastTime = advance.nextTime_s;
    this.info.submittedTime_s = this.lastTime;
    this.info.simulatedTime_s = this.lastTime;
    this.info.simulationLag_s = advance.lag_s;
    this.info.lastDt_s = dt;
    this.info.lastSubsteps = 1;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    const activeBodies = bodies.slice(0, 12);
    this.rigidSystem.syncBodies(activeBodies);
    const c = this.scene.container;
    const inflow = this.scene.fluid.inflow;
    const strength = inflow ? averageInflowStrength(inflow, this.lastTime - dt, this.lastTime) : 0;
    if (this.inflowBoundary && strength > 0) {
      const cellVolume = c.width_m * c.height_m * c.depth_m
        / (this.info.nx * this.info.ny * this.info.nz);
      this.referenceVolumeCells += this.inflowBoundary.flowRate_m3_s * strength * dt / cellVolume;
    }
    this.info.referenceLiquidVolume_cells = this.referenceVolumeCells;
    const outlet = this.inflowBoundary?.outletCenter_m;
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([
      this.info.nx, this.info.ny, this.info.nz, dt,
      c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz, this.scene.fluid.gravity_m_s2.y,
      c.width_m, c.height_m, c.depth_m, sceneHasTerrain(this.scene) ? 1 : 0,
      this.scene.fluid.density_kg_m3, this.scene.fluid.dynamicViscosity_Pa_s, 1, 0,
      this.scene.fluid.surfaceTension_N_m, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodies.length, c.top === "open" ? 1 : 0,
      outlet?.x ?? 0, outlet?.y ?? 0, outlet?.z ?? 0, inflow?.radius_m ?? 0,
      inflow?.velocity_m_s.x ?? 0, inflow?.velocity_m_s.y ?? 0, inflow?.velocity_m_s.z ?? 0, this.inflowBoundary?.apertureScale ?? 0,
      strength, this.referenceVolumeCells, c.fillFraction * this.info.ny, 4,
    ]));
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference step" });
    encoder.clearBuffer(this.rigidExchange);
    {
      const pass = encoder.beginComputePass({ label: "Uniform occupancy and transport preparation" });
      pass.setPipeline(this.pipelines.occupancy); pass.setBindGroup(0, this.occupancyGroup);
      pass.dispatchWorkgroups(Math.ceil(this.info.nx / 8), Math.ceil(this.info.nz / 8), 1);
      pass.setPipeline(this.pipelines.transport); pass.setBindGroup(0, this.transportCurrentGroup);
      pass.dispatchWorkgroups(Math.ceil((this.info.nx + 2) / 4), Math.ceil((this.info.ny + 2) / 4), Math.ceil((this.info.nz + 2) / 4));
      pass.setPipeline(this.pipelines.fluxScales); pass.setBindGroup(0, this.fluxScaleGroup);
      pass.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
      pass.end();
    }
    this.run(encoder, "Uniform velocity prediction", this.pipelines.advect, this.advectGroup);
    {
      const pass = encoder.beginComputePass({ label: "Uniform predicted transport" });
      pass.setPipeline(this.pipelines.transport); pass.setBindGroup(0, this.transportPredictedGroup);
      pass.dispatchWorkgroups(Math.ceil((this.info.nx + 2) / 4), Math.ceil((this.info.ny + 2) / 4), Math.ceil((this.info.nz + 2) / 4));
      pass.end();
    }
    this.run(encoder, "Uniform reverse advection", this.pipelines.reverse, this.reverseGroup);
    this.run(encoder, "Uniform MacCormack correction", this.pipelines.correct, this.correctGroup);
    for (let iteration = 0; iteration < this.info.pressureIterations; iteration += 1) {
      this.run(encoder, "Uniform Jacobi pressure iteration", this.pipelines.jacobi,
        iteration % 2 === 0 ? this.jacobiABGroup : this.jacobiBAGroup);
    }
    this.run(encoder, "Uniform pressure projection", this.pipelines.project, this.projectGroup);
    if (activeBodies.length > 0) {
      this.run(encoder, "Uniform rigid-body coupling", this.pipelines.coupleRigid, this.rigidGroup);
      encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
      encoder.copyTextureToTexture({ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]);
      const cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz);
      this.rigidSystem.encode(encoder, dt, cellVolume, 1, c.height_m / this.info.ny);
    }
    // Anti-diffuse the conservative VOF without creating or deleting mass:
    // removed air-side density is scattered back to the interface through a
    // fixed-point accumulation before the global residual controller runs.
    encoder.clearBuffer(this.conditioningScratch);
    this.run(encoder, "Uniform interface sharpening", this.pipelines.sharpenCompute, this.sharpenComputeGroup);
    this.run(encoder, "Uniform conserved sharpening scatter", this.pipelines.sharpenScatter, this.sharpenScatterGroup);
    this.run(encoder, "Uniform conserved sharpening resolve", this.pipelines.sharpenResolve, this.sharpenResolveGroup);
    // Preserve the conservative field globally without a CPU readback. The
    // controller measures raw VOF mass, then changes only a one-cell surface
    // band. Its response is expressed per simulated second in the shader, so
    // choosing a smaller solver step does not make it more aggressive.
    encoder.clearBuffer(this.reductions);
    this.run(encoder, "Uniform volume measurement", this.pipelines.measureVolume, this.volumeMeasureGroup);
    this.run(encoder, "Uniform bounded volume correction", this.pipelines.correctVolume, this.volumeCorrectionGroup);
    encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);

    // The renderer gets a smooth fractional reconstruction; transport,
    // projection, and diagnostics retain the corrected conservative VOF.
    this.run(encoder, "Uniform presentation smoothing A", this.pipelines.smoothSurface, this.smoothSurfaceAGroup);
    this.run(encoder, "Uniform presentation smoothing B", this.pipelines.smoothSurface, this.smoothSurfaceBGroup);
    encoder.clearBuffer(this.reductions);
    this.run(encoder, "Uniform diagnostics reduction", this.pipelines.reduce, this.reductionGroup);
    this.device.queue.submit([encoder.finish()]);
    const submittedTime = this.lastTime;
    void this.device.queue.onSubmittedWorkDone().then(() => {
      if (!this.disposed) this.info.completedTime_s = Math.max(this.info.completedTime_s ?? 0, submittedTime);
    }).catch(() => {});
    return true;
  }

  async readStats(): Promise<GPUEulerianInfo> {
    if (this.disposed || this.readbackPending) return this.info;
    this.readbackPending = true;
    this.statsReadback ??= this.device.createBuffer({ label: "Uniform reference diagnostics readback", size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference diagnostics readback" });
    encoder.copyBufferToBuffer(this.reductions, 0, this.statsReadback, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.statsReadback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(this.statsReadback.getMappedRange().slice(0));
      const reference = Math.max(1, this.referenceVolumeCells);
      this.info.representedVolumeCellSum = words[0] / 2048;
      this.info.volumeCellSum = words[3] / 2048;
      this.info.representedVolumeDrift = (this.info.representedVolumeCellSum - reference) / reference;
      this.info.rawVolumeDrift = (this.info.volumeCellSum - reference) / reference;
      this.info.volumeDrift = this.info.rawVolumeDrift;
      this.info.volumeTelemetrySource = "dense-volume";
      this.info.front_m = -this.scene.container.width_m / 2
        + words[1] * this.scene.container.width_m / this.info.nx;
      this.info.frontTelemetrySource = "dense-volume";
      this.info.maxSpeed_m_s = new Float32Array(new Uint32Array([words[2]]).buffer)[0];
      return this.info;
    } finally {
      if (this.statsReadback.mapState === "mapped") this.statsReadback.unmap();
      this.readbackPending = false;
    }
  }

  applySceneUniforms(scene: SceneDescription): void {
    this.scene = scene;
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [this.info.nx, this.info.ny, this.info.nz])
      : undefined;
  }

  get rigidRenderBuffer(): GPUBuffer { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer { return this.rigidSystem.motionBuffer; }
  setSelectedRigidBody(index: number): void { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin, direction); }
  readRigidBodyPoses() { return this.rigidSystem.readPoses(); }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const texture of new Set([
      this.velocityA, this.velocityB, this.velocityC, this.velocityD,
      this.pressureA, this.pressureB, this.volumeA, this.volumeB,
      this.surfaceA, this.surfaceB,
      this.heightA, this.heightB, this.terrainTexture,
      this.transportA, this.transportB, this.fluxScales,
    ])) texture.destroy();
    this.params.destroy();
    this.reductions.destroy();
    this.conditioningScratch.destroy();
    this.rigidSystem.destroy();
    this.rigidExchange.destroy();
    this.statsReadback?.destroy();
  }
}
