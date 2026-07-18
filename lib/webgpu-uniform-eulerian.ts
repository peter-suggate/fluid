import { damBreakFractions } from "./initial-fluid";
import { boundingRadius, initializeRigidBodies, type RigidBodyState } from "./rigid-body";
import {
  decodeGPURigidLoad,
  categorizedGPUPhysicsTime_ms,
  emptyGPUPhysicsTimings,
  GPU_PHYSICS_TIMESTAMP_CAPACITY,
  GPU_RIGID_EXCHANGE_BYTES,
  legacyUniformComputeShader,
  type GPUEulerianInfo,
  type GPUPhysicsStageId,
  type GPUPhysicsTimingField,
  type GPURigidLoad,
  type GPUVelocityTransport,
  type GPUQuality
} from "./webgpu-eulerian";
import type { SceneDescription } from "./model";
import { createTallCellLayout } from "./tall-cell-grid";
import { planGPUAdvance } from "./tall-cell-diagnostics";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "./inflow-boundary";
import { quadtreeChebyshevSpectrum, WebGPUQuadtreeTallCellProjection, type QuadtreeTallCellProjectionOptions } from "./webgpu-quadtree-tall-cell";
import { WebGPUOctreeProjection, type OctreeProjectionOptions } from "./webgpu-octree";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import {
  WebGPUSecondaryParticleSystem,
  type SecondaryParticleSamplingSource
} from "./webgpu-secondary-particles";

export type UniformVelocityTransport = GPUVelocityTransport;
export interface WebGPUUniformEulerianOptions { pressureIterations?: number; velocityTransport?: UniformVelocityTransport; densitySharpening?: boolean; tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings>; quadtreeTallCells?: Partial<QuadtreeTallCellProjectionOptions>; octree?: Partial<OctreeProjectionOptions>; /** One-way escaped spray droplets. Initially enabled for octree. */ secondaryParticles?: boolean; secondaryParticleCapacity?: number; quadtreeRebuildTopology?: boolean; quadtreeRebuildIntervalSteps?: number; quadtreeTopologyStaleSteps?: number; /** Fully GPU-resident every-step topology regeneration (Algorithm 1); default on for uncoupled parallel preconditioners. */ quadtreeInlineRebuild?: boolean; deferPipelineCompilation?: boolean }

/** Readback-free CFL subdivision for the next quadtree frame. */
export function proactiveQuadtreeSubsteps(
  previousMaxSpeed_m_s: number,
  inflowSpeed_m_s: number,
  gravityMagnitude_m_s2: number,
  frameDt_s: number,
  minimumCellSize_m: number,
  maximumSubsteps = 64
) {
  const safeDt = Math.max(0, Number.isFinite(frameDt_s) ? frameDt_s : 0);
  const safeCell = Math.max(1e-9, Number.isFinite(minimumCellSize_m) ? minimumCellSize_m : 0);
  const residentBound = Math.max(0, previousMaxSpeed_m_s, inflowSpeed_m_s);
  const velocityBound = residentBound + Math.max(0, gravityMagnitude_m_s2) * safeDt;
  const required = Math.ceil(velocityBound * safeDt / safeCell);
  return Math.max(1, Math.min(Math.max(1, Math.floor(maximumSubsteps)), required));
}

/** Convert a stale-limit wait into actual missed 60 Hz presentation frames. */
export function quadtreeMissedFrames(wait_ms: number, frameBudget_ms = 1000 / 60) {
  if (!(wait_ms > 0) || !(frameBudget_ms > 0)) return 0;
  return Math.max(0, Math.ceil(wait_ms / frameBudget_ms) - 1);
}

/** Bounded exponential backoff while a valid previous topology stays live. */
export function quadtreeRebuildRetryDelay(failureCount: number) {
  if (!(failureCount > 0)) return 0;
  return Math.min(60, 2 ** Math.min(6, Math.floor(failureCount)));
}

const quadtreePressureLabel = (projection: WebGPUQuadtreeTallCellProjection) => projection.solver === "chebyshev"
  ? "Chebyshev-Jacobi · row parallel"
  : ({ ic0: "ICCG(0)", blockic: "CG + block ICCG(0)", jacobi: "CG + diagonal Jacobi", line: "CG + vertical line Jacobi", poly: "CG + polynomial Jacobi", mg: "CG + geometric multigrid" })[projection.preconditioner];
const quadtreePressureDescription = (projection: WebGPUQuadtreeTallCellProjection, pressureIterations: number, tolerance: number) => projection.solver === "chebyshev"
  ? `${quadtreePressureLabel(projection)} · ${projection.info.pressureIterationBudget ?? pressureIterations} fixed passes · spectrum [${quadtreeChebyshevSpectrum.lower}, ${quadtreeChebyshevSpectrum.upper}] · experimental`
  : `${quadtreePressureLabel(projection)} · ${projection.info.pressureIterationBudget ?? pressureIterations} encoded / ${projection.info.pressureIterationHardBudget ?? pressureIterations} hard · relative ${tolerance}`;

/** The main-branch cubic solver retained as an A/B reference backend. */
export class WebGPUUniformEulerianSolver {
  readonly info: GPUEulerianInfo;
  private velocityA: GPUTexture; private velocityB: GPUTexture;
  private velocityC: GPUTexture; private velocityD: GPUTexture;
  private pressureA: GPUTexture; private pressureB: GPUTexture;
  private volumeA: GPUTexture; private volumeB: GPUTexture;
  private heightA: GPUTexture; private heightB: GPUTexture; private terrainTexture: GPUTexture;
  private transportA: GPUTexture; private transportB: GPUTexture; private fluxScales: GPUTexture;
  private transportSampler: GPUSampler;
  private params: GPUBuffer; private reductionBuffer: GPUBuffer; private sharpenBuffer: GPUBuffer;
  private rigidBuffer: GPUBuffer; private rigidExchangeBuffer: GPUBuffer;
  private statsReadbackBuffer?: GPUBuffer;
  private rigidReadbackPool: Array<{ buffer: GPUBuffer; busy: boolean }> = [];
  private bindGroupLayout: GPUBindGroupLayout;
  private advectPipeline!: GPUComputePipeline; private reversePipeline!: GPUComputePipeline;
  private correctPipeline!: GPUComputePipeline; private jacobiPipeline!: GPUComputePipeline;
  private projectPipeline!: GPUComputePipeline; private rigidPipeline!: GPUComputePipeline; private relaxSolidPhiPipeline!: GPUComputePipeline;
  private reductionPipeline!: GPUComputePipeline;
  private buildTransportPipeline!: GPUComputePipeline; private buildFluxScalesPipeline!: GPUComputePipeline;
  private buildOccupancyPipeline!: GPUComputePipeline;
  private sharpenComputePipeline!: GPUComputePipeline; private sharpenScatterPipeline!: GPUComputePipeline; private sharpenResolvePipeline!: GPUComputePipeline;
  private shaderModule:GPUShaderModule;private pipelineLayout:GPUPipelineLayout;private prepPipelineLayout:GPUPipelineLayout;
  private advectGroup: GPUBindGroup; private reverseGroup: GPUBindGroup; private correctGroup: GPUBindGroup;
  private jacobiABGroup: GPUBindGroup;
  private jacobiBAGroup: GPUBindGroup; private projectGroup: GPUBindGroup;
  private rigidGroup: GPUBindGroup; private reductionGroup: GPUBindGroup; private solidPhiGroup?: GPUBindGroup;
  private occupancyGroup: GPUBindGroup; private transportFromCurrentGroup: GPUBindGroup;
  private sharpenComputeGroup: GPUBindGroup; private sharpenScatterGroup: GPUBindGroup; private sharpenResolveGroup: GPUBindGroup;
  private transportFromPredictedGroup?: GPUBindGroup;
  private querySet?: GPUQuerySet; private queryResolve?: GPUBuffer;
  private querySegments: Array<{ name: GPUPhysicsTimingField; start: number; end: number }> = [];
  private queryCount = 0; private lastTime = 0; private readbackPending = false;
  private wallTimingPending = false;
  private validationChecked = false;
  private readonly inflowBoundary?: InflowGridBoundary;
  private readonly velocityTransport: UniformVelocityTransport;
  private readonly densitySharpening: boolean;
  private readonly transportConservativeVolume: boolean;
  private quadtreeProjection?: WebGPUQuadtreeTallCellProjection;
  private octreeProjection?: WebGPUOctreeProjection;
  private secondaryParticleSystem?: WebGPUSecondaryParticleSystem;
  private secondaryParticleSamplingSource?: SecondaryParticleSamplingSource;
  private readonly retiredQuadtreeProjections = new Set<WebGPUQuadtreeTallCellProjection>();
  private quadtreeRebuildPending = false;
  private quadtreeReadyProjection?: WebGPUQuadtreeTallCellProjection;
  private quadtreeRebuildBlockedFrames = 0;
  private quadtreeBlockedSince_ms?: number;
  private quadtreeRebuildFallbackWarned = false;
  private quadtreeRebuildCompletedCount = 0;
  private quadtreeRebuildFailureCount = 0;
  private quadtreeRebuildRetrySteps = 0;
  private readonly rebuildQuadtreeEachStep: boolean;
  private quadtreeStepsSinceTopology = 0;
  private quadtreeStepsSinceKick = 0;
  private quadtreeLastBodies: RigidBodyState[] = [];
  private readonly quadtreeRebuildInterval: number;
  private readonly quadtreeTopologyStaleLimit: number;
  private readonly quadtreeInlineRebuild: boolean;
  private disposed = false;
  private baseAllocatedBytes = 0;

  constructor(
    private device: GPUDevice,
    readonly scene: SceneDescription,
    quality: GPUQuality,
    private onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformEulerianOptions = {}
  ) {
    const c = scene.container, matched = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, options.tallCellSettings);
    const nx = matched.nx, ny = matched.fineNy, nz = matched.nz;
    this.velocityTransport = options.velocityTransport ?? "maccormack";
    this.densitySharpening = options.densitySharpening ?? true;
    this.rebuildQuadtreeEachStep = options.quadtreeRebuildTopology ?? true;
    // Narita et al. Algorithm 1 evaluates and subdivides the quadtree on every
    // Advance_Step. A caller may still request a slower experimental cadence,
    // but the paper-faithful default is one rebuild per simulation step.
    this.quadtreeRebuildInterval = Math.max(1, Math.round(options.quadtreeRebuildIntervalSteps ?? 1));
    // W6 acceptance pipelines one cadence-1 rebuild across at most two steps;
    // zero staleness remains Algorithm 1's stretch goal once the complete pack
    // stays resident and no readback/upload handshake remains.
    this.quadtreeTopologyStaleLimit = Math.max(0, Math.round(options.quadtreeTopologyStaleSteps ?? 2));
    this.quadtreeInlineRebuild = options.quadtreeInlineRebuild ?? true;
    this.inflowBoundary=scene.fluid.inflow?createInflowGridBoundary(scene.fluid.inflow,scene.container,[nx,ny,nz]):undefined;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture = (format: GPUTextureFormat) => device.createTexture({ size: [nx, ny, nz], dimension: "3d", format, usage });
    this.velocityA = texture("rgba32float"); this.velocityB = texture("rgba32float");
    this.velocityC = this.velocityTransport === "maccormack" ? texture("rgba32float") : this.velocityA;
    this.velocityD = this.velocityTransport === "maccormack" ? texture("rgba32float") : this.velocityB;
    this.pressureA = texture("r32float"); this.pressureB = texture("r32float");
    this.volumeA = texture("r32float"); this.volumeB = texture("r32float");
    this.heightA = device.createTexture({ label: "Uniform column fallback A", size: [nx, nz], format: "r32float", usage });
    this.heightB = device.createTexture({ label: "Uniform column occupancy", size: [nx, nz], format: "r32float", usage });
    this.terrainTexture = device.createTexture({ label: "Uniform terrain heights", size: [nx, nz], format: "r32float", usage });
    // Filterable fp16 transport fields, padded with a zero shell so hardware
    // clamp-to-edge sampling still reads zero at solid wall faces.
    const transportTexture = (label: string) => device.createTexture({ label, size: [nx + 2, ny + 2, nz + 2], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.transportA = transportTexture("Uniform transport velocity A");
    this.transportB = this.velocityTransport === "maccormack" ? transportTexture("Uniform transport velocity B") : this.transportA;
    this.fluxScales = device.createTexture({ label: "Uniform volume flux scales", size: [nx, ny, nz], dimension: "3d", format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.transportSampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.params = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductionBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidBuffer = device.createBuffer({ size: 12 * 96, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.sharpenBuffer = device.createBuffer({ label: "Uniform sharpening deposits", size: nx * ny * nz * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.rigidExchangeBuffer = device.createBuffer({ size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (device.features.has("timestamp-query")) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: GPU_PHYSICS_TIMESTAMP_CAPACITY });
      this.queryResolve = device.createBuffer({ size: GPU_PHYSICS_TIMESTAMP_CAPACITY * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    this.shaderModule = device.createShaderModule({ label: "Fluid Lab uniform reference kernels", code: legacyUniformComputeShader });
    void this.shaderModule.getCompilationInfo().then((info) => {
      for (const message of info.messages) if (message.type === "error") console.error(`Uniform GPU WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    }).catch(()=>{/* Device loss is reported by the renderer. */});
    this.bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } },
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
      { binding: 21, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } }
    ] });
    // The main layout already carries four storage textures (the per-stage
    // limit), so the transport/flux-scale writers get their own layout.
    const prepLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
      { binding: 18, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "3d" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.prepPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [prepLayout] });
    if(!options.deferPipelineCompilation)this.createPipelinesSync();
    const pressureIterations = Math.max(8, Math.min(400, Math.round(options.pressureIterations ?? (quality === "balanced" ? 64 : quality === "high" ? 80 : 96))));
    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count, regularLayers: ny, maximumNeighborDelta: 0,
      gridKind: "uniform", cellSize_m: Math.max(c.width_m / nx, c.height_m / ny, c.depth_m / nz),
      pressureIterations, allocatedBytes: count * (this.velocityTransport === "maccormack" ? 88 : 56) + (nx + 2) * (ny + 2) * (nz + 2) * 8 * (this.velocityTransport === "maccormack" ? 2 : 1), quality, encodedSteps: 0, maximumTallCellHeight: 0,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0, simulationLag_s: 0
    };
    this.baseAllocatedBytes = this.info.allocatedBytes;
    this.initializeVolume();
    if (options.quadtreeTallCells) {
      const initialCouplingBodies = scene.rigidBodies.length > 0 ? initializeRigidBodies(scene.rigidBodies) : [];
      this.quadtreeProjection = new WebGPUQuadtreeTallCellProjection(device, scene, { nx, ny, nz }, { velocityIn: this.velocityB, velocityOut: this.velocityA, velocityScratch: this.velocityD, volume: this.volumeA }, {
        pressureIterations,
        relativeTolerance: scene.numerics.pressureRelativeTolerance,
        adaptivityStrength: options.quadtreeTallCells.adaptivityStrength ?? 1,
        maximumLeafSize: options.quadtreeTallCells.maximumLeafSize ?? 16,
        opticalDepthFraction: options.quadtreeTallCells.opticalDepthFraction ?? 0.25,
        ...options.quadtreeTallCells
      }, undefined, initialCouplingBodies.length > 0 ? { bodies: initialCouplingBodies, dynamic: !!onRigidLoads } : undefined,options.deferPipelineCompilation);
      this.applyQuadtreeInfo(this.quadtreeProjection, pressureIterations);
    } else if (options.octree) {
      this.octreeProjection = new WebGPUOctreeProjection(device, scene, { nx, ny, nz }, {
        velocityIn: this.velocityB, velocityOut: this.velocityA, velocityScratch: this.velocityD,
        rigidBodies: this.rigidBuffer, rigidExchange: this.rigidExchangeBuffer, terrain: this.terrainTexture
      }, {
        pressureIterations,
        maximumLeafSize: options.octree.maximumLeafSize ?? (quality === "balanced" ? 4 : 8),
        adaptivity: options.octree.adaptivity ?? 1,
        jacobiRelaxation: options.octree.jacobiRelaxation ?? 0.8,
        extrapolationSweeps: options.octree.extrapolationSweeps ?? 4,
        leafSolver: options.octree.leafSolver,
        pressureWarmStart: options.octree.pressureWarmStart
      }, options.deferPipelineCompilation);
      this.applyOctreeInfo(this.octreeProjection);
      if (options.secondaryParticles !== false) {
        this.secondaryParticleSamplingSource = {
          surfaceTexture: this.octreeProjection.levelSetTexture,
          velocityTexture: this.velocityA,
          columnBaseTexture: this.heightA,
          fieldLayout: "uniform",
          surfaceEncoding: "level-set"
        };
        this.secondaryParticleSystem = new WebGPUSecondaryParticleSystem(device, { nx, ny, nz }, {
          width_m: c.width_m,
          height_m: c.height_m,
          depth_m: c.depth_m,
          topOpen: c.top === "open",
          gravity_m_s2: scene.fluid.gravity_m_s2,
          randomSeed: scene.randomSeed
        }, this.secondaryParticleSamplingSource, options.secondaryParticleCapacity, options.deferPipelineCompilation);
        this.info.secondaryParticleCapacity = this.secondaryParticleSystem.renderSource.capacity;
        this.info.allocatedBytes += this.secondaryParticleSystem.allocatedBytes;
      }
    }
    // The octree's resident level set is the complete liquid state. Keep VOF
    // transport only for the uniform solver and quadtree catastrophe recovery.
    this.transportConservativeVolume = !this.octreeProjection;
    // Construct every shared-solve bind group only after the adaptive surface
    // exists. Both adaptive methods use their resident level set as the sole
    // liquid authority. A compatibility VOF texture stays bound at binding 4,
    // but the octree path leaves it dormant.
    const surfaceAuthority = this.adaptiveProjection?.levelSetTexture ?? this.volumeA;
    const prepGroup = (velocity: GPUTexture, transport: GPUTexture) => device.createBindGroup({ layout: prepLayout, entries: [
      { binding: 0, resource: velocity.createView() }, { binding: 4, resource: this.volumeA.createView() },
      { binding: 6, resource: { buffer: this.params } }, { binding: 16, resource: transport.createView() },
      { binding: 18, resource: this.fluxScales.createView() }, { binding: 20, resource: surfaceAuthority.createView() }
    ] });
    this.transportFromCurrentGroup = prepGroup(this.velocityA, this.transportA);
    if (this.velocityTransport === "maccormack") this.transportFromPredictedGroup = prepGroup(this.velocityC, this.transportB);
    // Advection groups read the column occupancy from heightB; heightA stays
    // zero for the renderer's uniform column-base fallback.
    this.occupancyGroup = this.group(this.velocityA, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB);
    this.advectGroup = this.velocityTransport === "maccormack"
      ? this.group(this.velocityA, this.velocityC, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityB, this.velocityD)
      : this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.reverseGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityB, this.transportB) : this.advectGroup;
    this.correctGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityC, this.velocityD) : this.advectGroup;
    this.jacobiABGroup = this.group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.jacobiBAGroup = this.group(this.velocityB, this.velocityA, this.pressureB, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA);
    const finalPressure = pressureIterations % 2 === 0 ? this.pressureA : this.pressureB;
    const sparePressure = pressureIterations % 2 === 0 ? this.pressureB : this.pressureA;
    this.projectGroup = this.group(this.velocityB, this.velocityA, finalPressure, sparePressure, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.rigidGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenComputeGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.sharpenScatterGroup = this.group(this.velocityA, this.velocityB, this.pressureB, this.pressureA, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenResolveGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.reductionGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    // Paper Sec 3.9.1 phi-s over either resident adaptive level set: the pass
    // aliases the idle uniform pressure slots (pressureIn = pre-pass copy in
    // pressureA, pressureOut = the level-set texture itself). The velocity and
    // volume outputs are bound but never written by relaxSolidPhi.
    if (this.adaptiveProjection) this.solidPhiGroup = this.group(this.velocityA, this.velocityD, this.pressureA, this.adaptiveProjection.levelSetTexture, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, this.volumeA);
  }

  private pipelineDescriptor(entryPoint:string,prep=false):GPUComputePipelineDescriptor{return{layout:prep?this.prepPipelineLayout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  private createPipelinesSync(){const pipeline=(entryPoint:string,prep=false)=>this.device.createComputePipeline(this.pipelineDescriptor(entryPoint,prep));this.advectPipeline=pipeline(this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection");this.reversePipeline=pipeline("reverseAdvection");this.correctPipeline=pipeline("correctAdvection");this.jacobiPipeline=pipeline("jacobi");this.projectPipeline=pipeline("project");this.rigidPipeline=pipeline("coupleRigid");this.relaxSolidPhiPipeline=pipeline("relaxSolidPhi");this.reductionPipeline=pipeline("reduceDiagnostics");this.buildOccupancyPipeline=pipeline("buildOccupancy");this.buildTransportPipeline=pipeline("buildTransport",true);this.buildFluxScalesPipeline=pipeline("buildFluxScales",true);this.sharpenComputePipeline=pipeline("sharpenCompute");this.sharpenScatterPipeline=pipeline("sharpenScatter");this.sharpenResolvePipeline=pipeline("sharpenResolve");}
  static async createAsync(device:GPUDevice,scene:SceneDescription,quality:GPUQuality,onRigidLoads:((loads:GPURigidLoad[])=>void)|undefined,options:WebGPUUniformEulerianOptions,onProgress:(label:string,completed:number,total:number)=>void){const adaptive=!!(options.quadtreeTallCells||options.octree),secondary=!!options.octree&&options.secondaryParticles!==false,total=(options.quadtreeTallCells?41:options.octree?33:14)+(secondary?2:0);onProgress(adaptive?"Building adaptive pressure topology":"Allocating uniform solver resources",0,total);await new Promise<void>(resolve=>setTimeout(resolve,0));const solver=new WebGPUUniformEulerianSolver(device,scene,quality,onRigidLoads,{...options,deferPipelineCompilation:true});await solver.initializePipelines(onProgress);return solver;}
  private async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void){
    const definitions=[
      ["Advect velocity",this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection",false],["Reverse advection","reverseAdvection",false],
      ["Correct advection","correctAdvection",false],["Relax pressure","jacobi",false],["Project velocity","project",false],
      ["Couple rigid bodies","coupleRigid",false],["Relax solid level set","relaxSolidPhi",false],["Reduce diagnostics","reduceDiagnostics",false],["Build occupancy","buildOccupancy",false],
      ["Build transport field","buildTransport",true],["Build flux scales","buildFluxScales",true],
      ["Sharpen density","sharpenCompute",false],["Scatter sharpened mass","sharpenScatter",false],["Resolve sharpened mass","sharpenResolve",false]
    ] as const,compiled:GPUComputePipeline[]=[];
    const projectionPipelineCount=this.quadtreeProjection?27:this.octreeProjection?19:0;
    const total=definitions.length+projectionPipelineCount+(this.secondaryParticleSystem?2:0);
    for(let index=0;index<definitions.length;index+=1){const [label,entryPoint,prep]=definitions[index];onProgress(label,index,total);compiled.push(await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint,prep)));onProgress(label,index+1,total);}
    this.advectPipeline=compiled[0];this.reversePipeline=compiled[1];this.correctPipeline=compiled[2];this.jacobiPipeline=compiled[3];this.projectPipeline=compiled[4];this.rigidPipeline=compiled[5];this.relaxSolidPhiPipeline=compiled[6];this.reductionPipeline=compiled[7];this.buildOccupancyPipeline=compiled[8];this.buildTransportPipeline=compiled[9];this.buildFluxScalesPipeline=compiled[10];this.sharpenComputePipeline=compiled[11];this.sharpenScatterPipeline=compiled[12];this.sharpenResolvePipeline=compiled[13];
    if(this.quadtreeProjection)await this.quadtreeProjection.initializePipelines((label,completed)=>onProgress(label,definitions.length+completed,total));
    else if(this.octreeProjection)await this.octreeProjection.initializePipelines((label,completed)=>onProgress(label,definitions.length+completed,total));
    if(this.secondaryParticleSystem)await this.secondaryParticleSystem.initializePipelines((label,completed)=>onProgress(label,definitions.length+projectionPipelineCount+completed,total));
  }

  get volumeTexture() { return this.volumeA; }
  // Rendering contours the smooth resident level set when the quadtree
  // projection maintains one; the flux-form VOF field is near-binary and its
  // 0.5 contour is quantized to cell scale. Diagnostics keep reading the VOF
  // field through volumeTexture.
  private get adaptiveProjection() { return this.quadtreeProjection ?? this.octreeProjection; }
  get surfaceFieldTexture() { return this.adaptiveProjection?.levelSetTexture ?? this.volumeA; }
  get columnBaseTexture() { return this.heightA; }
  get gridCellTexture() { return this.adaptiveProjection?.topologyTexture; }
  get velocityTexture() { return this.velocityA; }
  get secondaryParticles() { return this.secondaryParticleSystem?.renderSource; }
  get gridPressureSamplesTexture() { return this.adaptiveProjection?.pressureSamplesTexture; }
  get gridPressureTexture() { return this.adaptiveProjection?.pressureTexture; }
  get gridDivergenceTexture() { return this.adaptiveProjection?.divergenceTexture; }
  /** Instrumentation view: velocity after advection/forces and before quadtree projection. */
  get preProjectionVelocityTexture() { return this.velocityB; }

  private initializeVolume() {
    const { nx, ny, nz } = this.info, c = this.scene.container;
    const data = new Float32Array(nx * ny * nz), dam = damBreakFractions(c.fillFraction);
    const terrainHeights = terrainColumnHeights(this.scene, nx, nz), cellHeight = c.height_m / ny;
    let initialSum = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const aboveGround = (j + 0.5) * cellHeight > terrainHeights[i + nx * k];
      const fill = aboveGround && (this.scene.fluid.initialCondition === "dam-break"
        ? (i + .5) / nx <= dam.width && (j + .5) / ny <= dam.height && (k + .5) / nz <= dam.depth
        : (j + .5) / ny <= c.fillFraction);
      data[i + nx * (j + ny * k)] = fill ? 1 : 0; if (fill) initialSum += 1;
    }
    const terrainCells = new Float32Array(nx * nz);
    for (let index = 0; index < terrainCells.length; index++) terrainCells[index] = terrainHeights[index] / cellHeight;
    const terrainRowBytes = nx * 4, terrainPadded = Math.ceil(terrainRowBytes / 256) * 256;
    const terrainPacked = new Uint8Array(terrainPadded * nz), terrainSource = new Uint8Array(terrainCells.buffer);
    for (let k = 0; k < nz; k++) terrainPacked.set(terrainSource.subarray(terrainRowBytes * k, terrainRowBytes * (k + 1)), terrainPadded * k);
    this.device.queue.writeTexture({ texture: this.terrainTexture }, terrainPacked, { bytesPerRow: terrainPadded, rowsPerImage: nz }, { width: nx, height: nz });
    Object.assign(this.info, { initialVolumeCellSum: initialSum, volumeCellSum: initialSum, representedVolumeCellSum: initialSum, representedVolumeDrift: 0, volumeDrift: 0, rawVolumeDrift: 0, maxSpeed_m_s: 0, front_m: this.scene.fluid.initialCondition === "dam-break" ? -c.width_m / 2 + dam.width * c.width_m : c.width_m / 2 });
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * ny * nz), source = new Uint8Array(data.buffer);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) packed.set(source.subarray(rowBytes * (j + ny * k), rowBytes * (j + ny * k + 1)), padded * (j + ny * k));
    for (const texture of [this.volumeA, this.volumeB]) this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
  }

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture, heightIn: GPUTexture, heightOut: GPUTexture, predictedVelocity: GPUTexture = velocityIn, reversedVelocity: GPUTexture = velocityIn, transport: GPUTexture = this.transportA, surfaceIn: GPUTexture = this.adaptiveProjection?.levelSetTexture ?? volumeIn) {
    return this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [
      { binding: 0, resource: velocityIn.createView() }, { binding: 1, resource: velocityOut.createView() },
      { binding: 2, resource: pressureIn.createView() }, { binding: 3, resource: pressureOut.createView() },
      { binding: 4, resource: volumeIn.createView() }, { binding: 5, resource: volumeOut.createView() },
      { binding: 6, resource: { buffer: this.params } }, { binding: 7, resource: heightIn.createView() },
      { binding: 8, resource: heightOut.createView() }, { binding: 9, resource: { buffer: this.reductionBuffer } },
      { binding: 10, resource: { buffer: this.rigidBuffer } }, { binding: 11, resource: { buffer: this.rigidExchangeBuffer } },
      { binding: 12, resource: predictedVelocity.createView() }, { binding: 13, resource: reversedVelocity.createView() },
      { binding: 14, resource: transport.createView() }, { binding: 15, resource: this.transportSampler },
      { binding: 17, resource: this.fluxScales.createView() },
      { binding: 19, resource: { buffer: this.sharpenBuffer } },
      { binding: 20, resource: surfaceIn.createView() },
      { binding: 21, resource: this.terrainTexture.createView() }
    ] });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
  }
  private applyQuadtreeInfo(projection: WebGPUQuadtreeTallCellProjection, pressureIterations = this.info.pressureIterations) {
    const quadtree = projection.info;
    Object.assign(this.info, {
      gridKind: "quadtree-tall-cell",
      surfaceField: "levelset",
      volumeControl: true,
      referenceLiquidVolume_cells: projection.surfaceDiagnostics.referenceVolumeCells,
      phiInterfaceCellCount: projection.surfaceDiagnostics.interfaceCells,
      volumeCorrectionNormalSpeed_cells_s: projection.surfaceDiagnostics.correctionSpeed,
      volumeControlAgreeWeight: projection.surfaceDiagnostics.volumeControlAgreeWeight,
      pressureSolver: quadtreePressureDescription(projection, pressureIterations, Math.max(this.scene.numerics.pressureRelativeTolerance, 1e-4)),
      compressionRatio: quadtree.compressionRatio, activeCompressionRatio: quadtree.compressionRatio,
      activeSampleCount: quadtree.liquidDofCount,
      allocatedBytes: this.baseAllocatedBytes + quadtree.allocatedBytes,
      quadtreeLeafCount: quadtree.leafCount, quadtreePressureSampleCount: quadtree.pressureSampleCount,
      quadtreeLiquidDofCount: quadtree.liquidDofCount, quadtreeFaceCount: quadtree.faceCount, quadtreeMLSProjectionRowCount: quadtree.mlsProjectionRowCount,
      quadtreeOpticalLayerMode: quadtree.opticalLayerMode, quadtreeOpticalAlpha: quadtree.opticalAlpha,
      quadtreeOpticalMinimumCells: quadtree.opticalMinimumCells, quadtreeOpticalMaximumCells: quadtree.opticalMaximumCells,
      quadtreeTallSegmentCount: quadtree.tallSegmentCount, quadtreeGhostFaceCount: quadtree.ghostFaceCount,
      quadtreeMaximumNeighborRatio: quadtree.maximumNeighborRatio, quadtreeMaximumFluidScale: quadtree.maximumFluidScale,
      quadtreeLevelSetMismatchFraction: projection.levelSetMismatchFraction ?? 0,
      quadtreeCulledDebrisCells: projection.surfaceDiagnostics.culledDebrisCells,
      quadtreeVofReconciliationActive: projection.surfaceDiagnostics.reconciliationActive,
      quadtreeGPUConstruction_ms: quadtree.gpuConstruction_ms,
      quadtreeGPUConstructionKernel_ms: quadtree.gpuConstructionKernel_ms,
      quadtreeGPUSparsePack_ms: quadtree.gpuSparsePack_ms,
      quadtreeCPUTopologyPack_ms: quadtree.cpuTopologyPack_ms,
      quadtreeCPURedistance_ms: quadtree.cpuRedistance_ms,
      quadtreeCPUQuadtreeDecode_ms: quadtree.cpuQuadtreeDecode_ms,
      quadtreeCPUTallGrid_ms: quadtree.cpuTallGrid_ms,
      quadtreeCPUVariationalAssembly_ms: quadtree.cpuVariationalAssembly_ms,
      quadtreeCPUSystemPack_ms: quadtree.cpuSystemPack_ms,
      quadtreeCPUICFactorization_ms: quadtree.cpuICFactorization_ms,
      quadtreeCPUResourceUpload_ms: quadtree.cpuResourceUpload_ms,
      quadtreeTopologyReused: quadtree.topologyReused,
      quadtreeTopologyReuseCount: quadtree.topologyReuseCount,
      quadtreePressureIterationsUsed: quadtree.pressureIterationsUsed,
      quadtreePressureIterationBudget: quadtree.pressureIterationBudget,
      quadtreePressureIterationHardBudget: quadtree.pressureIterationHardBudget,
      quadtreePressureConverged: quadtree.pressureConverged,
      quadtreeVelocityClampCount: quadtree.velocityClampCount,
      quadtreeFactorLevelCount: quadtree.factorLevelCount,
      quadtreeMultigridLevelCount: quadtree.multigridLevelCount,
      quadtreeMultigridCoarsestDofs: quadtree.multigridCoarsestDofs,
      quadtreePressurePhaseTimings: quadtree.pressurePhaseTimings,
      quadtreeRebuildCadenceSteps: this.quadtreeRebuildInterval,
      // Report the effective path, not merely the preference: coupled and
      // host-factorized pressure variants cannot consume the resident pack.
      quadtreeInlineRebuild: this.quadtreeInlineRebuild && projection.canEncodeInlineRebuild,
      quadtreeTopologyStaleLimit: this.quadtreeTopologyStaleLimit,
      quadtreeTopologyStaleSteps: this.quadtreeStepsSinceTopology,
      quadtreeRebuildCompletedCount: this.quadtreeRebuildCompletedCount,
      quadtreeTopologyReadbackBytes: quadtree.topologyReadbackBytes
    });
  }
  private applyOctreeInfo(projection: WebGPUOctreeProjection) {
    const octree = projection.info;
    Object.assign(this.info, {
      gridKind: "octree",
      surfaceField: "levelset",
      volumeControl: true,
      pressureSolver: projection.pressureSolverLabel,
      compressionRatio: octree.compressionRatio,
      activeCompressionRatio: octree.compressionRatio,
      activeSampleCount: octree.liquidDofCount,
      allocatedBytes: this.baseAllocatedBytes + octree.allocatedBytes + (this.secondaryParticleSystem?.allocatedBytes ?? 0),
      secondaryParticleCapacity: this.secondaryParticleSystem?.renderSource.capacity,
      quadtreeLeafCount: octree.leafCount,
      quadtreePressureSampleCount: octree.pressureSampleCount,
      quadtreeLiquidDofCount: octree.liquidDofCount,
      quadtreeMaximumNeighborRatio: octree.maximumNeighborRatio,
      quadtreeMaximumFluidScale: octree.maximumFluidScale,
      quadtreePressureIterationsUsed: octree.pressureIterationsUsed,
      quadtreePressureIterationBudget: octree.pressureIterationBudget,
      quadtreePressureIterationHardBudget: octree.pressureIterationHardBudget,
      quadtreeInlineRebuild: true,
      quadtreeRebuildCadenceSteps: 1,
      quadtreeTopologyStaleLimit: 0,
      quadtreeTopologyStaleSteps: 0,
      quadtreeTopologyReadbackBytes: 0
    });
  }
  private timing(name: GPUPhysicsTimingField) {
    if (!this.querySet || this.queryCount + 2 > GPU_PHYSICS_TIMESTAMP_CAPACITY) return undefined;
    const segment = { name, start: this.queryCount++, end: this.queryCount++ }; this.querySegments.push(segment); return segment;
  }

  private statsReadback() {
    // readbackPending guarantees that this buffer is never copied while it is
    // mapped. Keep enough room for the fixed query resolve allocation so
    // regular telemetry does not create/destroy a MAP_READ resource at 30 Hz.
    return this.statsReadbackBuffer ??= this.device.createBuffer({
      label: "Uniform pooled statistics readback",
      size: 16 + GPU_PHYSICS_TIMESTAMP_CAPACITY * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  }

  private rigidReadback() {
    let slot = this.rigidReadbackPool.find((candidate) => !candidate.busy);
    if (!slot) {
      slot = {
        buffer: this.device.createBuffer({
          label: "Uniform pooled rigid exchange readback",
          // The upper half carries the quadtree variational K^T p record.
          // Both channels then share one pooled map instead of allocating a
          // second staging buffer for every adaptive step.
          size: 2 * GPU_RIGID_EXCHANGE_BYTES,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        }),
        busy: false
      };
      this.rigidReadbackPool.push(slot);
    }
    slot.busy = true;
    return slot;
  }

  /**
   * A rebuild may resolve between frame encoding and queue submission. Waiting
   * on the queue immediately is therefore insufficient: onSubmittedWorkDone
   * only covers work submitted before it was called. Retire on the following
   * animation frame, after the frame loop has submitted every command buffer
   * that could still reference the old projection, and then wait for the GPU.
   */
  private retireQuadtreeProjection(projection: WebGPUQuadtreeTallCellProjection) {
    this.retiredQuadtreeProjections.add(projection);
    const waitForSubmittedFrame = () => {
      void this.device.queue.onSubmittedWorkDone().catch(() => { /* Device loss invalidates resources first. */ }).finally(() => {
        if (this.retiredQuadtreeProjections.delete(projection)) projection.destroy();
      });
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(waitForSubmittedFrame);
    else setTimeout(waitForSubmittedFrame, 0);
  }

  private shouldKickQuadtreeRebuild() {
    return this.quadtreeStepsSinceTopology >= this.quadtreeRebuildInterval;
  }

  /**
   * Launch the next topology rebuild from the current resident GPU level set.
   * Surface transport is per-step, so topology construction never integrates
   * a multi-step dt with an end-of-interval velocity.
   */
  private kickQuadtreeRebuild() {
    const previous = this.quadtreeProjection;
    if (!previous || this.quadtreeRebuildPending) return;
    const rebuildStartedAt = performance.now();
    this.quadtreeRebuildPending = true;
    this.info.quadtreeRebuildPending = true;
    const bodiesAtKick = this.quadtreeLastBodies.map((body) => structuredClone(body));
    this.quadtreeStepsSinceKick = 0;
    void previous.rebuildFromState(bodiesAtKick).then((next) => {
      if (this.quadtreeBlockedSince_ms !== undefined) {
        const missedFrames = quadtreeMissedFrames(performance.now() - this.quadtreeBlockedSince_ms);
        this.quadtreeRebuildBlockedFrames += missedFrames;
        this.quadtreeBlockedSince_ms = undefined;
      }
      this.quadtreeRebuildFallbackWarned = false;
      this.quadtreeRebuildPending = false;
      this.quadtreeRebuildFailureCount = 0;
      this.quadtreeRebuildRetrySteps = 0;
      this.info.quadtreeRebuildPending = false;
      this.info.quadtreeRebuildWall_ms = performance.now() - rebuildStartedAt;
      this.info.quadtreeRebuildBlockedFrames = this.quadtreeRebuildBlockedFrames;
      if (this.disposed) { if (next !== previous) next.destroy(); return; }
      // Stage the finished projection; advanceTo applies it at the fixed
      // step boundary so the swap schedule depends only on step counts,
      // never on rebuild wall time (keeps stepping deterministic).
      this.quadtreeReadyProjection = next;
    }).catch((error) => {
      this.quadtreeBlockedSince_ms = undefined;
      this.quadtreeRebuildFallbackWarned = false;
      this.quadtreeRebuildPending = false;
      this.info.quadtreeRebuildPending = false;
      this.quadtreeRebuildFailureCount += 1;
      this.quadtreeRebuildRetrySteps = quadtreeRebuildRetryDelay(this.quadtreeRebuildFailureCount);
      console.error(`Quadtree tall-cell rebuild failed; reusing the previous topology and retrying in ${this.quadtreeRebuildRetrySteps} steps`, error);
    });
  }

  private applyReadyQuadtreeProjection() {
    const next = this.quadtreeReadyProjection, previous = this.quadtreeProjection;
    if (!next || !previous) return;
    this.quadtreeReadyProjection = undefined;
    this.quadtreeRebuildCompletedCount += 1;
    this.quadtreeProjection = next; this.applyQuadtreeInfo(next);
    // Steps advanced on the previous topology while the rebuild was in
    // flight; the swapped topology is stepsSinceKick steps behind the
    // surface, which the swap boundary keeps bounded.
    this.quadtreeStepsSinceTopology = this.quadtreeStepsSinceKick;
    // The replaced projection's buffers may still be referenced by queued
    // steps; only release them once the queue drains.
    if (next !== previous) this.retireQuadtreeProjection(previous);
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []) {
    if (this.disposed) return false;
    // Deterministic bounded-staleness rebuild pipeline. Algorithm 1 wants the
    // quadtree constructed before advection and pressure, but a synchronous
    // handshake costs one full GPU-readback + worker-pack round trip per
    // step. Instead, up to quadtreeTopologyStaleLimit steps run ahead on the
    // previous topology while its replacement is assembled, and the finished
    // projection is applied exactly at that step boundary — blocking there if
    // the rebuild is still in flight — so the swap schedule depends only on
    // step counts, never on rebuild wall time. refreshFaces re-derives the
    // free-surface fractions from the live level set every solve, so only
    // the DOF layout is stale in between.
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep && (this.quadtreeRebuildPending || this.quadtreeReadyProjection) && this.quadtreeStepsSinceKick >= this.quadtreeTopologyStaleLimit) {
      if (!this.quadtreeReadyProjection) {
        this.quadtreeBlockedSince_ms ??= performance.now();
        this.info.quadtreeRebuildBlockedFrames = this.quadtreeRebuildBlockedFrames;
        const blocked_ms = performance.now() - this.quadtreeBlockedSince_ms;
        // A rebuild is an optimization of the pressure layout, not a reason
        // to starve surface transport indefinitely. After three presentation
        // budgets, keep advancing on the previous valid topology until the
        // asynchronous replacement arrives.
        // A failed replacement already proved that waiting cannot make this
        // attempt usable. Advance immediately on the previous projection so
        // retry backoff can count down; only an in-flight first attempt gets
        // the short presentation-budget grace period.
        if (this.quadtreeRebuildFailureCount === 0 && blocked_ms < 3 * 1000 / 60) return false;
        if (!this.quadtreeRebuildFallbackWarned) {
          console.warn(`Quadtree topology rebuild blocked for ${blocked_ms.toFixed(1)} ms; reusing the previous topology until it completes`);
          this.quadtreeRebuildFallbackWarned = true;
        }
      }
      if (this.quadtreeReadyProjection) this.applyReadyQuadtreeProjection();
    }
    const advance = planGPUAdvance(time_s, this.lastTime, this.scene.numerics.maxDt_s); if (!advance) return false;
    const delta = advance.dt_s; if (delta < 1e-6) { this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; return true; }
    this.lastTime = advance.nextTime_s; this.info.submittedTime_s = this.lastTime; this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; const c = this.scene.container, rho = this.scene.fluid.density_kg_m3, sigma = this.scene.fluid.surfaceTension_N_m;
    // Proactive CFL control. The latest completed reduction is the previous
    // projected maximum; gravity is the only unbounded explicit acceleration
    // before the next solve, so prevMax + |g| dt is a conservative readback-
    // free bound for choosing this frame's subdivisions.
    const hMin = Math.min(c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz);
    const inflowSpeed = this.scene.fluid.inflow ? Math.hypot(this.scene.fluid.inflow.velocity_m_s.x, this.scene.fluid.inflow.velocity_m_s.y, this.scene.fluid.inflow.velocity_m_s.z) : 0;
    const substeps = this.adaptiveProjection ? proactiveQuadtreeSubsteps(
      this.info.maxSpeed_m_s ?? 0,
      inflowSpeed,
      Math.hypot(this.scene.fluid.gravity_m_s2.x, this.scene.fluid.gravity_m_s2.y, this.scene.fluid.gravity_m_s2.z),
      delta,
      hMin
    ) : 1;
    const dt = delta / substeps; this.info.lastDt_s = dt; this.info.lastSubsteps = substeps;
    const activeBodies = bodies.slice(0, 12), bodyData = new Float32Array(12 * 24), shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
    activeBodies.forEach((body, index) => { const o = index * 24, d = body.description.dimensions_m, q = body.orientation; bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, shapeIndex[body.description.shape], d.x, d.y, d.z, boundingRadius(body), q.w, q.x, q.y, q.z, body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z, 0, body.angularVelocity_rad_s.x, body.angularVelocity_rad_s.y, body.angularVelocity_rad_s.z, body.description.density_kg_m3, rho * body.inverseMass_kg, rho * body.inverseInertiaBody_kg_m2.x, rho * body.inverseInertiaBody_kg_m2.y, rho * body.inverseInertiaBody_kg_m2.z], o); });
    this.device.queue.writeBuffer(this.rigidBuffer, 0, bodyData); this.info.encodedSteps = (this.info.encodedSteps ?? 0) + substeps;
    this.octreeProjection?.setCouplingBodies(activeBodies.length, activeBodies.some((body) => body.inverseMass_kg > 0));
    const inflow=this.scene.fluid.inflow,outlet=this.inflowBoundary?.outletCenter_m,inflowStepStrength=inflow?averageInflowStrength(inflow,this.lastTime-delta,this.lastTime):0;
    if(this.adaptiveProjection&&this.inflowBoundary){const cellVolume=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);this.adaptiveProjection.addSurfaceReferenceVolumeCells(this.inflowBoundary.flowRate_m3_s*inflowStepStrength*delta/cellVolume);}
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([this.info.nx, this.info.ny, this.info.nz, dt, c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz, this.scene.fluid.gravity_m_s2.y, c.width_m, c.height_m, c.depth_m, sceneHasTerrain(this.scene) ? 1 : 0, rho, this.scene.fluid.dynamicViscosity_Pa_s, this.transportConservativeVolume ? 1 : 0, this.adaptiveProjection ? 1 : 0, sigma, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodies.length, c.top === "open" ? 1 : 0,outlet?.x??0,outlet?.y??0,outlet?.z??0,inflow?.radius_m??0,inflow?.velocity_m_s.x??0,inflow?.velocity_m_s.y??0,inflow?.velocity_m_s.z??0,this.inflowBoundary?.apertureScale??0,inflowStepStrength,0,0,0]));
    if(this.secondaryParticleSystem&&this.secondaryParticleSamplingSource)this.secondaryParticleSystem.prepareStep(dt,this.secondaryParticleSamplingSource);
    this.querySegments = []; this.queryCount = 0; if (!this.validationChecked) this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label: "Uniform GPU fluid step" }), totalTiming = this.timing("total_ms");
    if (totalTiming && this.querySet) { const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: totalTiming.start } }); pass.end(); }
    encoder.clearBuffer(this.rigidExchangeBuffer);
    // Narita Algorithm 1: regenerate the quadtree at the top of every step.
    // The fully GPU-resident rebuild encodes ahead of advection in the same
    // command stream (queue order = algorithm order) with zero staleness;
    // the asynchronous pipeline below remains the warmup/regrow/rigid path.
    let inlineRebuildEncoded = false;
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep && this.quadtreeInlineRebuild && !this.quadtreeRebuildPending && !this.quadtreeReadyProjection && this.quadtreeProjection.canEncodeInlineRebuild) {
      const timing = this.timing("layerConstruction_ms");
      if (timing && this.querySet) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.start } }); marker.end(); }
      inlineRebuildEncoded = this.quadtreeProjection.encodeInlineRebuild(encoder);
      if (timing && this.querySet) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.end } }); marker.end(); }
    } else if (this.octreeProjection) {
      const timing = this.timing("layerConstruction_ms");
      if (timing && this.querySet) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.start } }); marker.end(); }
      inlineRebuildEncoded = this.octreeProjection.encodeInlineRebuild(encoder);
      if (timing && this.querySet) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.end } }); marker.end(); }
    }
    for (let substep = 0; substep < substeps; substep += 1) {
      // The first rebuild was encoded above so topology is ready before any
      // dynamics. If CFL control subdivides this advance, phi moves after each
      // projection; rebuild again before the next substep so a newly exposed
      // interface can never remain inside a coarse pressure leaf.
      if (substep > 0 && this.octreeProjection) this.octreeProjection.encodeInlineRebuild(encoder);
      {
        const timing = this.timing("advection_ms");
        const paddedWorkgroups: [number, number, number] = [Math.ceil((this.info.nx + 2) / 4), Math.ceil((this.info.ny + 2) / 4), Math.ceil((this.info.nz + 2) / 4)];
        // Occupancy, transport extrapolation, and flux scales only read the
        // projected state, so they share one pass ahead of the predictor.
        const prep = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start } } : undefined);
        prep.setPipeline(this.buildOccupancyPipeline); prep.setBindGroup(0, this.occupancyGroup);
        prep.dispatchWorkgroups(Math.ceil(this.info.nx / 8), Math.ceil(this.info.nz / 8), 1);
        prep.setPipeline(this.buildTransportPipeline); prep.setBindGroup(0, this.transportFromCurrentGroup);
        prep.dispatchWorkgroups(...paddedWorkgroups);
        if (this.transportConservativeVolume) {
          prep.setPipeline(this.buildFluxScalesPipeline);
          prep.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
        }
        prep.end();
        const predict = encoder.beginComputePass(timing && this.querySet && this.velocityTransport === "semi-lagrangian" ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.end } } : undefined);
        this.dispatch(predict, this.advectPipeline, this.advectGroup); predict.end();
        if (this.velocityTransport === "maccormack" && this.transportFromPredictedGroup) {
          const predictedTransport = encoder.beginComputePass();
          predictedTransport.setPipeline(this.buildTransportPipeline); predictedTransport.setBindGroup(0, this.transportFromPredictedGroup);
          predictedTransport.dispatchWorkgroups(...paddedWorkgroups); predictedTransport.end();
          const reverse = encoder.beginComputePass(); this.dispatch(reverse, this.reversePipeline, this.reverseGroup); reverse.end();
          const correct = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(correct, this.correctPipeline, this.correctGroup); correct.end();
        }
      }
      if (this.densitySharpening && this.transportConservativeVolume) {
        // Mass-Conserving Eulerian Liquid Simulation Sec 3.5: sharpen the
        // advected density before the pressure solve. volumeB -> volumeA
        // (sharpened, deltas in pressureB) -> volumeB (resolved deposits).
        const timing = this.timing("conditioning_ms");
        encoder.clearBuffer(this.sharpenBuffer);
        const computePass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start } } : undefined); this.dispatch(computePass, this.sharpenComputePipeline, this.sharpenComputeGroup); computePass.end();
        const scatterPass = encoder.beginComputePass(); this.dispatch(scatterPass, this.sharpenScatterPipeline, this.sharpenScatterGroup); scatterPass.end();
        const resolvePass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(resolvePass, this.sharpenResolvePipeline, this.sharpenResolveGroup); resolvePass.end();
      }
      if (this.adaptiveProjection) {
        if (this.transportConservativeVolume) encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        const surfaceInflow = inflow && this.inflowBoundary ? {
          outletCenter_m: this.inflowBoundary.outletCenter_m, radius_m: inflow.radius_m,
          velocity_m_s: inflow.velocity_m_s, apertureScale: this.inflowBoundary.apertureScale,
          strength: inflowStepStrength
        } : undefined;
        const timestampWrites = (timing: { start: number; end: number } | undefined) => timing && this.querySet ? {
          querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end
        } : undefined;
        if (this.octreeProjection) {
          this.octreeProjection.encode(
            encoder,
            this.info.nx,
            this.info.ny,
            this.info.nz,
            timestampWrites(this.timing("pressure_ms")),
            {
              projection: timestampWrites(this.timing("projection_ms")),
              extrapolation: this.octreeProjection.extrapolationSweepCount > 0 ? timestampWrites(this.timing("extrapolation_ms")) : undefined,
              materialization: timestampWrites(this.timing("materialization_ms"))
            }
          );
        } else if (this.quadtreeProjection) {
          this.quadtreeProjection.encode(encoder, this.info.nx, this.info.ny, this.info.nz, timestampWrites(this.timing("pressure_ms")));
        }
        // Transport phi from the freshly projected, narrow-band-extrapolated
        // velocity. Sampling the previous frame here was the one-frame lag
        // that froze crests and newly exposed interface cells.
        const surfaceTiming = this.timing("surfaceUpdate_ms");
        this.adaptiveProjection.encodeSurface(encoder, dt, surfaceInflow, this.scene.numerics.maxDt_s, surfaceTiming && this.querySet ? {
          querySet: this.querySet, beginningOfPassWriteIndex: surfaceTiming.start, endOfPassWriteIndex: surfaceTiming.end
        } : undefined);
      } else {
        { const timing = this.timing("pressure_ms"); for (let iteration = 0; iteration < this.info.pressureIterations; iteration += 1) { const first = iteration === 0, last = iteration === this.info.pressureIterations - 1; const pass = encoder.beginComputePass(timing && this.querySet && (first || last) ? { timestampWrites: { querySet: this.querySet, ...(first ? { beginningOfPassWriteIndex: timing.start } : {}), ...(last ? { endOfPassWriteIndex: timing.end } : {}) } } : undefined); this.dispatch(pass, this.jacobiPipeline, iteration % 2 === 0 ? this.jacobiABGroup : this.jacobiBAGroup); pass.end(); } }
        { const timing = this.timing("projection_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.projectPipeline, this.projectGroup); pass.end(); }
      }
      // Chentanez & Müller Sec. 3.9.1 runs for every pressure backend. Both
      // adaptive projections constrain the normal face flux variationally;
      // this per-substep Brinkman blend supplies tangential drag and interior
      // momentum for moving bodies. Octree additionally folds the dynamic
      // body's pressure response into the next presentation batch, avoiding a
      // global K^T p reduction at every pressure iterate. Phi-s relaxation
      // keeps the resident level set sane inside either backend's solids so
      // they displace water instead of carrying sealed liquid plugs.
      if (activeBodies.length > 0) {
        if (this.adaptiveProjection && this.solidPhiGroup) {
          encoder.copyTextureToTexture({ texture: this.adaptiveProjection.levelSetTexture }, { texture: this.pressureA }, [this.info.nx, this.info.ny, this.info.nz]);
          const phiPass = encoder.beginComputePass(); this.dispatch(phiPass, this.relaxSolidPhiPipeline, this.solidPhiGroup); phiPass.end();
        }
        const timing = this.timing("rigidCoupling_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined);
        this.dispatch(pass, this.rigidPipeline, this.rigidGroup); pass.end();
        if (this.transportConservativeVolume) encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        encoder.copyTextureToTexture({ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]);
      }
      if (this.secondaryParticleSystem) {
        const sprayTiming = this.timing("spray_ms");
        this.secondaryParticleSystem.encode(encoder, sprayTiming && this.querySet ? {
          querySet: this.querySet,
          beginningOfPassWriteIndex: sprayTiming.start,
          endOfPassWriteIndex: sprayTiming.end
        } : undefined);
      }
    }
    encoder.clearBuffer(this.reductionBuffer); { const timing = this.timing("diagnostics_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.reductionPipeline, this.reductionGroup); pass.end(); }
    if (totalTiming && this.querySet) { const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: totalTiming.end } }); pass.end(); }
    if (this.querySet && this.queryResolve && this.queryCount > 0) encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.queryResolve, 0);
    let exchangeReadback: { buffer: GPUBuffer; busy: boolean } | undefined;
    const quadtreeImpulseProjection = this.quadtreeProjection;
    let hasQuadtreeImpulseReadback = false;
    if (activeBodies.length > 0 && this.onRigidLoads) {
      exchangeReadback = this.rigidReadback();
      encoder.copyBufferToBuffer(this.rigidExchangeBuffer, 0, exchangeReadback.buffer, 0, GPU_RIGID_EXCHANGE_BYTES);
      // The blend reaction and the variational constraint impulse are
      // sequential operator splits over the same interval; both channels are
      // read together and summed per body below.
      hasQuadtreeImpulseReadback = !!quadtreeImpulseProjection?.encodeBodyImpulseReadback(encoder, exchangeReadback.buffer, GPU_RIGID_EXCHANGE_BYTES);
    }
    const submittedAt = performance.now(); this.device.queue.submit([encoder.finish()]);
    // Keep the iteration-budget EMA fed even when nobody polls stats: the
    // solve feedback readback is 48 bytes, pending-gated, and non-blocking, so
    // the encoded CG budget tracks iterations-to-tolerance at step rhythm.
    void this.quadtreeProjection?.readSolveDiagnostics();
    if (this.octreeProjection && inlineRebuildEncoded) {
      for (let rebuild = 0; rebuild < substeps; rebuild += 1) this.octreeProjection.finishInlineRebuild();
      this.applyOctreeInfo(this.octreeProjection);
      this.info.quadtreeRebuildCompletedCount = (this.info.quadtreeRebuildCompletedCount ?? 0) + substeps;
    }
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep) {
      this.quadtreeStepsSinceTopology += 1; this.quadtreeStepsSinceKick += 1;
      if (inlineRebuildEncoded) {
        // The step just submitted carries its own freshly regenerated
        // topology: staleness is zero by construction and every step counts
        // as a completed rebuild (paper cadence).
        this.quadtreeProjection.finishInlineRebuild();
        this.quadtreeStepsSinceTopology = 0;
        this.quadtreeRebuildCompletedCount += 1;
        // Republish projection telemetry: the non-blocking packControl
        // monitor refreshes leaf/DOF/face counts without any swap.
        this.applyQuadtreeInfo(this.quadtreeProjection);
      }
      this.info.quadtreeTopologyStaleSteps = this.quadtreeStepsSinceTopology;
      this.info.quadtreeTopologyStaleLimit = inlineRebuildEncoded ? 0 : this.quadtreeTopologyStaleLimit;
      this.quadtreeLastBodies = activeBodies;
      if (this.quadtreeRebuildRetrySteps > 0) this.quadtreeRebuildRetrySteps -= 1;
      if (!inlineRebuildEncoded && this.quadtreeRebuildRetrySteps === 0 && !this.quadtreeRebuildPending && !this.quadtreeReadyProjection && this.shouldKickQuadtreeRebuild()) this.kickQuadtreeRebuild();
    }
    if (!this.wallTimingPending) { this.wallTimingPending = true; void this.device.queue.onSubmittedWorkDone().then(() => { this.info.gpuQueueWall_ms = performance.now() - submittedAt; this.info.gpuQueueSimulation_s = delta; }).catch(() => { /* Device loss is handled by the renderer. */ }).finally(() => { this.wallTimingPending = false; }); }
    if (exchangeReadback) {
      const slot = exchangeReadback, readback = slot.buffer, elapsed = delta, cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz);
      const mapPromise = readback.mapAsync(GPUMapMode.READ);
      void mapPromise.then(() => {
        // WebGPU mapped ranges may not overlap. Decode the fixed dense record
        // and the variational record from their disjoint halves.
        const words = new Int32Array(readback.getMappedRange(0, GPU_RIGID_EXCHANGE_BYTES));
        const impulses = hasQuadtreeImpulseReadback && quadtreeImpulseProjection
          ? quadtreeImpulseProjection.decodeMappedBodyImpulseReadback(readback, GPU_RIGID_EXCHANGE_BYTES)
          : [];
        const loads = activeBodies.map((body, index) => decodeGPURigidLoad(body.description.id, words, index, elapsed, cellVolume, substeps));
        readback.unmap();
        // Both channels cover the same interval, so impulses add per body and
        // the interval stays `delta`; the variational displaced volume (from
        // the voxelized [A] field) is the better of the two estimates.
        const impulseById = new Map(impulses.map((impulse) => [impulse.bodyId, impulse]));
        this.onRigidLoads?.(loads.map((load) => {
          const extra = impulseById.get(load.bodyId);
          if (!extra) return load;
          return {
            ...load,
            impulse_N_s: { x: load.impulse_N_s.x + extra.impulse_N_s.x, y: load.impulse_N_s.y + extra.impulse_N_s.y, z: load.impulse_N_s.z + extra.impulse_N_s.z },
            angularImpulse_N_m_s: { x: load.angularImpulse_N_m_s.x + extra.angularImpulse_N_m_s.x, y: load.angularImpulse_N_m_s.y + extra.angularImpulse_N_m_s.y, z: load.angularImpulse_N_m_s.z + extra.angularImpulse_N_m_s.z },
            displacedVolume_m3: extra.displacedVolume_m3 > 0 ? extra.displacedVolume_m3 : load.displacedVolume_m3
          };
        }));
      }).catch(async () => {
        await mapPromise.catch(() => { /* Device loss is handled by the renderer. */ });
        if (readback.mapState === "mapped") readback.unmap();
      }).finally(() => { slot.busy = false; });
    }
    if (!this.validationChecked) { this.validationChecked = true; void this.device.popErrorScope().then((error) => { if (error) console.error(`Uniform GPU validation: ${error.message}`); }).catch(() => { /* Device loss is handled by the renderer. */ }); }
    return true;
  }

  async readStats() {
    if ((this.info.encodedSteps ?? 0) === 0 || this.readbackPending) return this.info;
    this.readbackPending = true; const quadtreeDiagnostics = this.adaptiveProjection?.readSolveDiagnostics(); const surfaceDiagnosticsPromise = this.adaptiveProjection?.readSurfaceDiagnostics(); const querySegments = this.querySegments, queryBytes = this.queryResolve ? this.queryCount * 8 : 0;
    const buffer = this.statsReadback(), encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.reductionBuffer, 0, buffer, 0, 16); if (this.queryResolve && queryBytes > 0) encoder.copyBufferToBuffer(this.queryResolve, 0, buffer, 16, queryBytes);
    this.device.queue.submit([encoder.finish()]);
    const mapPromise = buffer.mapAsync(GPUMapMode.READ);
    try {
      const [, , surfaceDiagnostics] = await Promise.all([mapPromise, quadtreeDiagnostics, surfaceDiagnosticsPromise]);
    if (this.quadtreeProjection) this.info.quadtreeVelocityClampCount = this.quadtreeProjection.info.velocityClampCount ?? 0;
    const words = new Uint32Array(buffer.getMappedRange(0, 16)), initial = Math.max(1, this.info.initialVolumeCellSum ?? 1);
    const conservativeVolumeCells=words[3]/2048;this.info.rawVolumeDrift=this.transportConservativeVolume?(conservativeVolumeCells-initial)/initial:undefined;
    if(surfaceDiagnostics){const reference=Math.max(1,surfaceDiagnostics.referenceVolumeCells);this.info.referenceLiquidVolume_cells=surfaceDiagnostics.referenceVolumeCells;this.info.volumeCellSum=surfaceDiagnostics.volumeCells;this.info.representedVolumeCellSum=surfaceDiagnostics.volumeCells;this.info.volumeDrift=(surfaceDiagnostics.volumeCells-reference)/reference;this.info.representedVolumeDrift=this.info.volumeDrift;this.info.phiInterfaceCellCount=surfaceDiagnostics.interfaceCells;this.info.volumeCorrectionNormalSpeed_cells_s=surfaceDiagnostics.correctionSpeed;this.info.volumeControlAgreeWeight=surfaceDiagnostics.volumeControlAgreeWeight;this.info.quadtreeCulledDebrisCells=surfaceDiagnostics.culledDebrisCells;this.info.quadtreeLevelSetMismatchFraction=surfaceDiagnostics.mismatchFraction;this.info.quadtreeVofReconciliationActive=surfaceDiagnostics.reconciliationActive;}
    else{this.info.representedVolumeCellSum=words[0]/2048;this.info.representedVolumeDrift=(this.info.representedVolumeCellSum-initial)/initial;this.info.volumeCellSum=conservativeVolumeCells;this.info.volumeDrift=this.info.rawVolumeDrift;}
    this.info.front_m = -this.scene.container.width_m / 2 + words[1] * this.scene.container.width_m / this.info.nx;
    this.info.maxSpeed_m_s = new Float32Array(new Uint32Array([words[2]]).buffer)[0];
    if (this.quadtreeProjection?.relativeResidual !== undefined) this.info.pressureRelativeResidual = this.quadtreeProjection.relativeResidual;
    if (this.quadtreeProjection?.residualRms !== undefined) this.info.pressureResidual = this.quadtreeProjection.residualRms;
    if (this.quadtreeProjection) {
      this.info.quadtreePressureIterationsUsed = this.quadtreeProjection.info.pressureIterationsUsed;
      this.info.quadtreePressureIterationBudget = this.quadtreeProjection.info.pressureIterationBudget;
      this.info.quadtreePressureIterationHardBudget = this.quadtreeProjection.info.pressureIterationHardBudget;
      this.info.quadtreePressureConverged = this.quadtreeProjection.info.pressureConverged;
      this.info.quadtreeFactorLevelCount = this.quadtreeProjection.info.factorLevelCount;
      this.info.quadtreeMultigridLevelCount = this.quadtreeProjection.info.multigridLevelCount;
      this.info.quadtreeMultigridCoarsestDofs = this.quadtreeProjection.info.multigridCoarsestDofs;
      this.info.quadtreePressurePhaseTimings = this.quadtreeProjection.info.pressurePhaseTimings;
      this.info.pressureSolver = quadtreePressureDescription(this.quadtreeProjection, this.info.pressureIterations, Math.max(this.scene.numerics.pressureRelativeTolerance, 1e-4));
    } else if (this.octreeProjection) {
      this.info.quadtreePressureIterationsUsed = this.octreeProjection.info.pressureIterationsUsed;
      this.info.quadtreePressureIterationBudget = this.octreeProjection.info.pressureIterationBudget;
      this.info.quadtreePressureIterationHardBudget = this.octreeProjection.info.pressureIterationHardBudget;
      this.info.pressureSolver = this.octreeProjection.pressureSolverLabel;
    }
    if (queryBytes > 0) {
      const times = new BigUint64Array(buffer.getMappedRange(16, queryBytes));
      const stageByField: Partial<Record<GPUPhysicsTimingField, GPUPhysicsStageId>> = { preparation_ms: "preparation", layerConstruction_ms: "topology", advection_ms: "advection", conditioning_ms: "conditioning", remeshing_ms: "remeshing", pressure_ms: "pressure", projection_ms: "projection", extrapolation_ms: "extrapolation", materialization_ms: "materialization", surfaceUpdate_ms: "surfaceUpdate", rigidCoupling_ms: "rigidCoupling", spray_ms: "spray", diagnostics_ms: "diagnostics" };
      const activeStages = [...new Set(querySegments.map((segment) => stageByField[segment.name]).filter((stage): stage is GPUPhysicsStageId => Boolean(stage)))];
      const timings = emptyGPUPhysicsTimings(activeStages);
      for (const segment of querySegments) timings[segment.name] += Math.max(0, Number(times[segment.end] - times[segment.start])) / 1e6;
      const categorized = categorizedGPUPhysicsTime_ms(timings);
      /* Empty marker passes may collapse to one timestamp on Metal. Never publish a total smaller than its directly timed real passes. */
      timings.total_ms = Math.max(timings.total_ms, categorized); timings.overhead_ms = Math.max(0, timings.total_ms - categorized); this.info.gpuTimings = timings; this.info.gpuStep_ms = timings.total_ms;
    }
      return this.info;
    } finally {
      // A diagnostic promise can reject before mapAsync settles. The pooled
      // staging buffer cannot be copied again while it is pending or mapped.
      await mapPromise.catch(() => { /* Device loss is handled by the renderer. */ });
      if (buffer.mapState === "mapped") buffer.unmap();
      this.readbackPending = false;
    }
  }

  destroy() {
    this.disposed = true;
    if (this.quadtreeReadyProjection && this.quadtreeReadyProjection !== this.quadtreeProjection) this.quadtreeReadyProjection.destroy();
    this.quadtreeReadyProjection = undefined;
    this.quadtreeProjection?.destroySharedSurface();
    this.quadtreeProjection?.destroy();
    this.octreeProjection?.destroy();
    this.secondaryParticleSystem?.destroy();
    for (const projection of this.retiredQuadtreeProjections) projection.destroy();
    this.retiredQuadtreeProjections.clear();
    for (const texture of new Set([this.velocityA, this.velocityB, this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB, this.transportA, this.transportB, this.fluxScales])) texture.destroy();
    this.params.destroy(); this.reductionBuffer.destroy(); this.sharpenBuffer.destroy(); this.rigidBuffer.destroy(); this.rigidExchangeBuffer.destroy(); this.statsReadbackBuffer?.destroy(); for (const slot of this.rigidReadbackPool) slot.buffer.destroy(); this.querySet?.destroy(); this.queryResolve?.destroy();
  }
}
