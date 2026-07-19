import { combineInitialBrickWet, damBreakFractions, initialFluidBrickContainsCell } from "./initial-fluid";
import { initializeRigidBodies, type RigidBodyState } from "./rigid-body";
import {
  categorizedGPUPhysicsTime_ms,
  emptyGPUPhysicsTimings,
  GPU_PHYSICS_TIMESTAMP_CAPACITY,
  GPU_RIGID_EXCHANGE_BYTES,
  legacyUniformComputeShader,
  retiredBulkFluxScaleClearShader,
  retiredBulkTransportClearShader,
  retiredBulkVelocityClearShader,
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
import type { SparseSurfaceBandGPUSource } from "./webgpu-sparse-surface-band";
import type { FluidBrickAtlasSamplingSource } from "./webgpu-brick-atlas";
import { FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES, FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES } from "./webgpu-fluid-brick-residency";
import { WebGPURigidBodySystem } from "./webgpu-rigid-body";

export type UniformVelocityTransport = GPUVelocityTransport;
export interface WebGPUUniformEulerianOptions { pressureIterations?: number; velocityTransport?: UniformVelocityTransport; densitySharpening?: boolean; tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings>; quadtreeTallCells?: Partial<QuadtreeTallCellProjectionOptions>; octree?: Partial<OctreeProjectionOptions>; /** A/B switch; sparse is used only by safe octree mirror sources. */ brickSparseVelocityAdvection?: boolean; /** Independently dispatch current/predicted transport preparation over the safe bulk worklist. */ brickSparseTransportPreparation?: boolean; /** Independently reduce column occupancy and conservative flux scales over the safe bulk worklist. */ brickSparseOccupancyFluxPreparation?: boolean; /** Escaped spray droplets. Initially enabled for octree. */ secondaryParticles?: boolean; secondaryParticleCapacity?: number; /** Bounded near-interface particle-to-level-set correction; zero keeps spray strictly one-way. */ secondaryParticleSurfaceCorrection?: number; quadtreeRebuildTopology?: boolean; quadtreeRebuildIntervalSteps?: number; quadtreeTopologyStaleSteps?: number; /** Fully GPU-resident every-step topology regeneration (Algorithm 1); default on for uncoupled parallel preconditioners. */ quadtreeInlineRebuild?: boolean; deferPipelineCompilation?: boolean }

/** Explicit capillary-wave stability bound for a finest cell. */
export function capillaryStableDt_s(
  density_kg_m3: number,
  surfaceTension_N_m: number,
  minimumCellSize_m: number,
  safety = 0.5
) {
  const density = Number.isFinite(density_kg_m3) ? Math.max(1e-9, density_kg_m3) : 1e-9;
  const sigma = Number.isFinite(surfaceTension_N_m) ? Math.max(0, surfaceTension_N_m) : 0;
  const h = Number.isFinite(minimumCellSize_m) ? Math.max(1e-9, minimumCellSize_m) : 1e-9;
  const boundedSafety = Number.isFinite(safety) ? Math.max(0.05, Math.min(1, safety)) : 0.5;
  return sigma > 0 ? boundedSafety * Math.sqrt(density * h * h * h / (Math.PI * sigma)) : Number.POSITIVE_INFINITY;
}

/** Readback-free CFL and capillary subdivision for the next adaptive frame. */
export function proactiveQuadtreeSubsteps(
  previousMaxSpeed_m_s: number,
  inflowSpeed_m_s: number,
  gravityMagnitude_m_s2: number,
  frameDt_s: number,
  minimumCellSize_m: number,
  maximumSubsteps = 64,
  density_kg_m3 = 1_000,
  surfaceTension_N_m = 0
) {
  const safeDt = Math.max(0, Number.isFinite(frameDt_s) ? frameDt_s : 0);
  const safeCell = Math.max(1e-9, Number.isFinite(minimumCellSize_m) ? minimumCellSize_m : 0);
  const residentBound = Math.max(0, previousMaxSpeed_m_s, inflowSpeed_m_s);
  const velocityBound = residentBound + Math.max(0, gravityMagnitude_m_s2) * safeDt;
  const cflRequired = Math.ceil(velocityBound * safeDt / safeCell);
  const capillaryDt = capillaryStableDt_s(density_kg_m3, surfaceTension_N_m, safeCell);
  const capillaryRequired = Number.isFinite(capillaryDt) && capillaryDt > 0 ? Math.ceil(safeDt / capillaryDt) : 1;
  const required = Math.max(cflRequired, capillaryRequired);
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
  private params: GPUBuffer; private reductionBuffer: GPUBuffer; private sharpenBuffer: GPUBuffer; private occupancyColumns: GPUBuffer; private occupancyCell64Dispatch: GPUBuffer;
  private bulkAtlasFallbackPageTable: GPUBuffer; private bulkAtlasFallbackParams: GPUBuffer;
  private bulkAtlasFallbackVelocity: GPUTexture; private bulkAtlasFallbackWorklist: GPUBuffer;
  private bulkAtlasControlSampleSparse: GPUBuffer; private bulkAtlasControlSampleDense: GPUBuffer; private bulkAtlasControlSparse: GPUBuffer; private bulkAtlasControlOff: GPUBuffer;
  private rigidBuffer: GPUBuffer; private rigidExchangeBuffer: GPUBuffer;
  private rigidSystem: WebGPURigidBodySystem;
  private statsReadbackBuffer?: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private advectPipeline!: GPUComputePipeline; private reversePipeline!: GPUComputePipeline;
  private correctPipeline!: GPUComputePipeline; private jacobiPipeline!: GPUComputePipeline;
  private projectPipeline!: GPUComputePipeline; private rigidPipeline!: GPUComputePipeline; private relaxSolidPhiPipeline!: GPUComputePipeline;
  private reductionPipeline!: GPUComputePipeline;
  private buildTransportPipeline!: GPUComputePipeline; private buildFluxScalesPipeline!: GPUComputePipeline;
  private buildOccupancyPipeline!: GPUComputePipeline; private buildSparseOccupancyPipeline!: GPUComputePipeline; private resolveSparseOccupancyPipeline!: GPUComputePipeline;
  private retiredVelocityClearPipeline: GPUComputePipeline; private retiredVelocityClearLayout: GPUBindGroupLayout;
  private retiredVelocityClearGroups: GPUBindGroup[] = [];
  private retiredTransportClearPipeline: GPUComputePipeline; private retiredTransportClearLayout: GPUBindGroupLayout;
  private retiredTransportClearGroups: GPUBindGroup[] = [];
  private retiredFluxScaleClearPipeline: GPUComputePipeline; private retiredFluxScaleClearLayout: GPUBindGroupLayout;
  private retiredFluxScaleClearGroup: GPUBindGroup;
  private sharpenComputePipeline!: GPUComputePipeline; private sharpenScatterPipeline!: GPUComputePipeline; private sharpenResolvePipeline!: GPUComputePipeline;
  private shaderModule:GPUShaderModule;private pipelineLayout:GPUPipelineLayout;private prepPipelineLayout:GPUPipelineLayout;
  private advectGroup: GPUBindGroup; private reverseGroup: GPUBindGroup; private correctGroup: GPUBindGroup;
  private jacobiABGroup: GPUBindGroup;
  private jacobiBAGroup: GPUBindGroup; private projectGroup: GPUBindGroup;
  private rigidGroup: GPUBindGroup; private reductionGroup: GPUBindGroup; private solidPhiGroup?: GPUBindGroup;
  private occupancyGroup: GPUBindGroup; private transportFromCurrentGroup: GPUBindGroup; private fluxScaleGroup: GPUBindGroup;
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
  private readonly sparseVelocityAdvectionRequested: boolean;
  private readonly sparseTransportPreparationRequested: boolean;
  private readonly sparseOccupancyFluxPreparationRequested: boolean;
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
    this.sparseVelocityAdvectionRequested = options.brickSparseVelocityAdvection ?? true;
    // Kept as an explicit A/B until a sparse-domain scene proves a win. A
    // full-footprint ocean retains most bulk bricks, and the indirect decode
    // currently offsets the saved empty-air stores.
    this.sparseTransportPreparationRequested = options.brickSparseTransportPreparation ?? false;
    this.sparseOccupancyFluxPreparationRequested = options.brickSparseOccupancyFluxPreparation ?? false;
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
    this.bulkAtlasFallbackPageTable = device.createBuffer({ label: "Disabled bulk atlas page table", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.bulkAtlasFallbackPageTable, 0, new Uint32Array([0xffff_ffff]));
    this.bulkAtlasFallbackParams = device.createBuffer({ label: "Disabled bulk atlas parameters", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bulkAtlasFallbackVelocity = device.createTexture({ label: "Disabled bulk atlas velocity", size: [1, 1, 1], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.bulkAtlasFallbackWorklist = device.createBuffer({ label: "Disabled bulk velocity worklist", size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.bulkAtlasControlSampleSparse = device.createBuffer({ label: "Enable bulk atlas sampling and sparse targets", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bulkAtlasControlSampleDense = device.createBuffer({ label: "Enable bulk atlas sampling with dense targets", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bulkAtlasControlSparse = device.createBuffer({ label: "Enable sparse bulk targets", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bulkAtlasControlOff = device.createBuffer({ label: "Disable bulk atlas sampling", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.bulkAtlasControlSampleSparse, 0, new Uint32Array([1, 1, 0, 0]));
    device.queue.writeBuffer(this.bulkAtlasControlSampleDense, 0, new Uint32Array([1, 0, 0, 0]));
    device.queue.writeBuffer(this.bulkAtlasControlSparse, 0, new Uint32Array([0, 1, 0, 0]));
    this.reductionBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.sharpenBuffer = device.createBuffer({ label: "Uniform sharpening deposits", size: nx * ny * nz * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.occupancyColumns = device.createBuffer({ label: "Sparse column occupancy maxima", size: this.sparseOccupancyFluxPreparationRequested ? nx * nz * 4 : 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Dawn/Metal requires the indirect args to be disjoint from the worklist
    // simultaneously bound as storage. Stage the producer's byte-48 ABI once
    // per refresh; payload decoding still reads the original resident list.
    this.occupancyCell64Dispatch = device.createBuffer({ label: "Sparse occupancy/flux cell64 dispatch", size: 12, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.rigidExchangeBuffer = device.createBuffer({ size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidSystem = new WebGPURigidBodySystem(device, scene, this.rigidExchangeBuffer, this.terrainTexture);
    this.rigidBuffer = this.rigidSystem.stateBuffer;
    this.rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
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
      ,{ binding: 22, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 23, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
      ,{ binding: 24, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
      ,{ binding: 25, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
      ,{ binding: 26, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 27, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    // The main layout already carries four storage textures (the per-stage
    // limit), so the transport/flux-scale writers get their own layout.
    const prepLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
      { binding: 18, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "3d" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 24, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 25, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 26, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.prepPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [prepLayout] });
    this.retiredVelocityClearLayout = device.createBindGroupLayout({ label: "Retired bulk velocity clear layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const retiredClearModule = device.createShaderModule({ label: "Retired bulk velocity clear", code: retiredBulkVelocityClearShader });
    this.retiredVelocityClearPipeline = device.createComputePipeline({ label: "Clear retired bulk velocities", layout: device.createPipelineLayout({ bindGroupLayouts: [this.retiredVelocityClearLayout] }), compute: { module: retiredClearModule, entryPoint: "clearRetiredVelocity" } });
    this.retiredTransportClearLayout = device.createBindGroupLayout({ label: "Retired bulk transport clear layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const retiredTransportClearModule = device.createShaderModule({ label: "Retired bulk transport clear", code: retiredBulkTransportClearShader });
    this.retiredTransportClearPipeline = device.createComputePipeline({ label: "Clear retired bulk transport fields", layout: device.createPipelineLayout({ bindGroupLayouts: [this.retiredTransportClearLayout] }), compute: { module: retiredTransportClearModule, entryPoint: "clearRetiredTransport" } });
    this.retiredFluxScaleClearLayout = device.createBindGroupLayout({ label: "Retired bulk flux-scale clear layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const retiredFluxScaleClearModule = device.createShaderModule({ label: "Retired bulk flux-scale clear", code: retiredBulkFluxScaleClearShader });
    this.retiredFluxScaleClearPipeline = device.createComputePipeline({ label: "Clear retired bulk flux scales", layout: device.createPipelineLayout({ bindGroupLayouts: [this.retiredFluxScaleClearLayout] }), compute: { module: retiredFluxScaleClearModule, entryPoint: "clearRetiredFluxScales" } });
    if(!options.deferPipelineCompilation)this.createPipelinesSync();
    const pressureIterations = Math.max(8, Math.min(400, Math.round(options.pressureIterations ?? (quality === "balanced" ? 64 : quality === "high" ? 80 : 96))));
    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count, regularLayers: ny, maximumNeighborDelta: 0,
      gridKind: "uniform", cellSize_m: Math.max(c.width_m / nx, c.height_m / ny, c.depth_m / nz),
      pressureIterations, allocatedBytes: count * (this.velocityTransport === "maccormack" ? 88 : 56) + (nx + 2) * (ny + 2) * (nz + 2) * 8 * (this.velocityTransport === "maccormack" ? 2 : 1) + this.occupancyColumns.size + this.occupancyCell64Dispatch.size, quality, encodedSteps: 0, maximumTallCellHeight: 0,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0, simulationLag_s: 0
    };
    this.baseAllocatedBytes = this.info.allocatedBytes;
    this.initializeVolume();
    if (options.quadtreeTallCells) {
      // Dynamic bodies are consumed from the resident storage buffer by the
      // dense immersed-boundary pass. Only immutable bodies enter the CPU-built
      // variational topology; otherwise a moving GPU body would leave behind a
      // stale host-authored K matrix and force a pose readback to rebuild it.
      const initialCouplingBodies = scene.rigidBodies.filter((body) => body.motion === "static").map((body) => initializeRigidBodies([body])[0]);
      this.quadtreeProjection = new WebGPUQuadtreeTallCellProjection(device, scene, { nx, ny, nz }, { velocityIn: this.velocityB, velocityOut: this.velocityA, velocityScratch: this.velocityD, volume: this.volumeA }, {
        pressureIterations,
        relativeTolerance: scene.numerics.pressureRelativeTolerance,
        adaptivityStrength: options.quadtreeTallCells.adaptivityStrength ?? 1,
        maximumLeafSize: options.quadtreeTallCells.maximumLeafSize ?? 16,
        opticalDepthFraction: options.quadtreeTallCells.opticalDepthFraction ?? 0.25,
        ...options.quadtreeTallCells
      }, undefined, initialCouplingBodies.length > 0 ? { bodies: initialCouplingBodies, dynamic: false } : undefined,options.deferPipelineCompilation);
      this.applyQuadtreeInfo(this.quadtreeProjection, pressureIterations);
    } else if (options.octree) {
      this.octreeProjection = new WebGPUOctreeProjection(device, scene, { nx, ny, nz }, {
        velocityIn: this.velocityB, velocityOut: this.velocityA, velocityScratch: this.velocityD,
        rigidBodies: this.rigidBuffer, rigidExchange: this.rigidExchangeBuffer, terrain: this.terrainTexture
      }, {
        pressureIterations,
        maximumLeafSize: options.octree.maximumLeafSize ?? 16,
        adaptivity: options.octree.adaptivity ?? 1,
        interfaceRefinementBandCells: options.octree.interfaceRefinementBandCells ?? 4,
        surfaceDetailStrength: options.octree.surfaceDetailStrength ?? 0,
        sparseSurfaceBand: options.octree.sparseSurfaceBand ?? "off",
        surfaceRefinementFactor: options.octree.surfaceRefinementFactor ?? 2,
        sparseSurfaceBandCells: options.octree.sparseSurfaceBandCells ?? 4,
        sparseSurfacePageFraction: options.octree.sparseSurfacePageFraction ?? 0.75,
        brickAtlas: options.octree.brickAtlas ?? "mirror",
        brickPreActivation: options.octree.brickPreActivation ?? true,
        brickSparseSurface: options.octree.brickSparseSurface ?? true,
        brickSparseExtrapolation: options.octree.brickSparseExtrapolation ?? false,
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
          density_kg_m3: scene.fluid.density_kg_m3,
          surfaceTension_N_m: scene.fluid.surfaceTension_N_m,
          randomSeed: scene.randomSeed
        }, this.secondaryParticleSamplingSource, options.secondaryParticleCapacity, options.deferPipelineCompilation, options.secondaryParticleSurfaceCorrection ?? 0);
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
    const bulkSource = this.octreeProjection?.fluidBrickAtlasSamplingSource;
    const sparseTransportSource = this.sparseTransportPreparationRequested
      && bulkSource?.mode === "mirror" && bulkSource.sparseDispatchSafe ? bulkSource : undefined;
    const sparseOccupancyFluxSource = this.sparseOccupancyFluxPreparationRequested
      && bulkSource?.mode === "mirror" && bulkSource.sparseDispatchSafe ? bulkSource : undefined;
    const prepGroup = (velocity: GPUTexture, transport: GPUTexture, sparseSource?: FluidBrickAtlasSamplingSource) => device.createBindGroup({ layout: prepLayout, entries: [
      { binding: 0, resource: velocity.createView() }, { binding: 4, resource: this.volumeA.createView() },
      { binding: 6, resource: { buffer: this.params } }, { binding: 16, resource: transport.createView() },
      { binding: 18, resource: this.fluxScales.createView() }, { binding: 20, resource: surfaceAuthority.createView() },
      { binding: 24, resource: { buffer: sparseSource?.params ?? this.bulkAtlasFallbackParams } },
      { binding: 25, resource: { buffer: sparseSource ? this.bulkAtlasControlSparse : this.bulkAtlasControlOff } },
      { binding: 26, resource: { buffer: sparseSource?.bulkWorklist ?? this.bulkAtlasFallbackWorklist } }
    ] });
    this.transportFromCurrentGroup = prepGroup(this.velocityA, this.transportA, sparseTransportSource);
    if (this.velocityTransport === "maccormack") this.transportFromPredictedGroup = prepGroup(this.velocityC, this.transportB, sparseTransportSource);
    this.fluxScaleGroup = prepGroup(this.velocityA, this.transportA, sparseOccupancyFluxSource);
    // Advection groups read the column occupancy from heightB; heightA stays
    // zero for the renderer's uniform column-base fallback.
    this.occupancyGroup = this.group(this.velocityA, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB, this.velocityA, this.velocityA, this.transportA, surfaceAuthority, false, "occupancy");
    this.advectGroup = this.velocityTransport === "maccormack"
      ? this.group(this.velocityA, this.velocityC, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityB, this.velocityD, this.transportA, surfaceAuthority, true, "velocity")
      : this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, surfaceAuthority, true, "velocity");
    this.reverseGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityB, this.transportB, surfaceAuthority, false, "velocity") : this.advectGroup;
    // The uniform path samples its current VOF field as the liquid authority.
    // Its correction output is volumeA, so sampling volumeA in the same
    // dispatch would alias one texture as both sampled and writable.
    const correctionSurfaceAuthority = this.adaptiveProjection?.levelSetTexture ?? this.volumeB;
    this.correctGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityC, this.velocityD, this.transportA, correctionSurfaceAuthority, true, "velocity") : this.advectGroup;
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
    const clearWorklist = bulkSource?.mode === "mirror" && bulkSource.sparseDispatchSafe ? bulkSource.bulkWorklist : this.bulkAtlasFallbackWorklist;
    const clearParams = bulkSource?.mode === "mirror" && bulkSource.sparseDispatchSafe ? bulkSource.params : this.bulkAtlasFallbackParams;
    this.retiredVelocityClearGroups = [this.velocityA, this.velocityB, this.velocityC, this.velocityD].map((velocity) => device.createBindGroup({
      label: "Retired bulk velocity clear bindings",
      layout: this.retiredVelocityClearLayout,
      entries: [
        { binding: 0, resource: velocity.createView() },
        { binding: 1, resource: { buffer: clearWorklist } },
        { binding: 2, resource: { buffer: clearParams } },
      ],
    }));
    this.retiredTransportClearGroups = [...new Set([this.transportA, this.transportB])].map((transport) => device.createBindGroup({
      label: "Retired bulk transport clear bindings",
      layout: this.retiredTransportClearLayout,
      entries: [
        { binding: 0, resource: transport.createView() },
        { binding: 1, resource: { buffer: clearWorklist } },
        { binding: 2, resource: { buffer: clearParams } },
      ],
    }));
    this.retiredFluxScaleClearGroup = device.createBindGroup({
      label: "Retired bulk flux-scale clear bindings",
      layout: this.retiredFluxScaleClearLayout,
      entries: [
        { binding: 0, resource: this.fluxScales.createView() },
        { binding: 1, resource: { buffer: clearWorklist } },
        { binding: 2, resource: { buffer: clearParams } },
      ],
    });
    // Paper Sec 3.9.1 phi-s over either resident adaptive level set: the pass
    // aliases the idle uniform pressure slots (pressureIn = pre-pass copy in
    // pressureA, pressureOut = the level-set texture itself). The velocity and
    // volume outputs are bound but never written by relaxSolidPhi.
    if (this.adaptiveProjection) this.solidPhiGroup = this.group(this.velocityA, this.velocityD, this.pressureA, this.adaptiveProjection.levelSetTexture, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, this.volumeA);
    if (this.octreeProjection && !options.deferPipelineCompilation) this.publishInitialSparseScene();
  }

  private pipelineDescriptor(entryPoint:string,prep=false):GPUComputePipelineDescriptor{return{layout:prep?this.prepPipelineLayout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  private createPipelinesSync(){const pipeline=(entryPoint:string,prep=false)=>this.device.createComputePipeline(this.pipelineDescriptor(entryPoint,prep));this.advectPipeline=pipeline(this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection");this.reversePipeline=pipeline("reverseAdvection");this.correctPipeline=pipeline("correctAdvection");this.jacobiPipeline=pipeline("jacobi");this.projectPipeline=pipeline("project");this.rigidPipeline=pipeline("coupleRigid");this.relaxSolidPhiPipeline=pipeline("relaxSolidPhi");this.reductionPipeline=pipeline("reduceDiagnostics");this.buildOccupancyPipeline=pipeline("buildOccupancy");this.buildSparseOccupancyPipeline=pipeline("buildSparseOccupancy");this.resolveSparseOccupancyPipeline=pipeline("resolveSparseOccupancy");this.buildTransportPipeline=pipeline("buildTransport",true);this.buildFluxScalesPipeline=pipeline("buildFluxScales",true);this.sharpenComputePipeline=pipeline("sharpenCompute");this.sharpenScatterPipeline=pipeline("sharpenScatter");this.sharpenResolvePipeline=pipeline("sharpenResolve");}
  static async createAsync(device:GPUDevice,scene:SceneDescription,quality:GPUQuality,onRigidLoads:((loads:GPURigidLoad[])=>void)|undefined,options:WebGPUUniformEulerianOptions,onProgress:(label:string,completed:number,total:number)=>void){const adaptive=!!(options.quadtreeTallCells||options.octree),secondary=!!options.octree&&options.secondaryParticles!==false,secondaryPipelines=secondary?2+((options.secondaryParticleSurfaceCorrection??0)>0?3:0):0,total=(options.quadtreeTallCells?43:options.octree?35:16)+secondaryPipelines;onProgress(adaptive?"Building adaptive pressure topology":"Allocating uniform solver resources",0,total);await new Promise<void>(resolve=>setTimeout(resolve,0));const solver=new WebGPUUniformEulerianSolver(device,scene,quality,onRigidLoads,{...options,deferPipelineCompilation:true});await solver.initializePipelines(onProgress);return solver;}
  private async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void){
    const definitions=[
      ["Advect velocity",this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection",false],["Reverse advection","reverseAdvection",false],
      ["Correct advection","correctAdvection",false],["Relax pressure","jacobi",false],["Project velocity","project",false],
      ["Couple rigid bodies","coupleRigid",false],["Relax solid level set","relaxSolidPhi",false],["Reduce diagnostics","reduceDiagnostics",false],["Build occupancy","buildOccupancy",false],
      ["Build sparse occupancy","buildSparseOccupancy",false],["Resolve sparse occupancy","resolveSparseOccupancy",false],
      ["Build transport field","buildTransport",true],["Build flux scales","buildFluxScales",true],
      ["Sharpen density","sharpenCompute",false],["Scatter sharpened mass","sharpenScatter",false],["Resolve sharpened mass","sharpenResolve",false]
    ] as const,compiled:GPUComputePipeline[]=[];
    const projectionPipelineCount=this.quadtreeProjection?27:this.octreeProjection?19:0;
    const total=definitions.length+projectionPipelineCount+(this.secondaryParticleSystem?.pipelineCount??0);
    for(let index=0;index<definitions.length;index+=1){const [label,entryPoint,prep]=definitions[index];onProgress(label,index,total);compiled.push(await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint,prep)));onProgress(label,index+1,total);}
    this.advectPipeline=compiled[0];this.reversePipeline=compiled[1];this.correctPipeline=compiled[2];this.jacobiPipeline=compiled[3];this.projectPipeline=compiled[4];this.rigidPipeline=compiled[5];this.relaxSolidPhiPipeline=compiled[6];this.reductionPipeline=compiled[7];this.buildOccupancyPipeline=compiled[8];this.buildSparseOccupancyPipeline=compiled[9];this.resolveSparseOccupancyPipeline=compiled[10];this.buildTransportPipeline=compiled[11];this.buildFluxScalesPipeline=compiled[12];this.sharpenComputePipeline=compiled[13];this.sharpenScatterPipeline=compiled[14];this.sharpenResolvePipeline=compiled[15];
    if(this.quadtreeProjection)await this.quadtreeProjection.initializePipelines((label,completed)=>onProgress(label,definitions.length+completed,total));
    else if(this.octreeProjection)await this.octreeProjection.initializePipelines((label,completed)=>onProgress(label,definitions.length+completed,total));
    if(this.secondaryParticleSystem)await this.secondaryParticleSystem.initializePipelines((label,completed)=>onProgress(label,definitions.length+projectionPipelineCount+completed,total));
    if (this.octreeProjection) this.publishInitialSparseScene();
  }

  /** Publish a complete t=0 scene after rigid-solid raster pipelines exist. */
  private publishInitialSparseScene() {
    if (!this.octreeProjection) return;
    const initialSparseScene = this.device.createCommandEncoder({ label: "Publish initial sparse-brick scene" });
    this.octreeProjection.encodeSurfaceBand(initialSparseScene, 0);
    this.octreeProjection.encodeInlineRebuild(initialSparseScene);
    this.octreeProjection.encodeOverlayMaterialization(initialSparseScene);
    this.octreeProjection.encodeSparseBrickWorld(initialSparseScene);
    this.device.queue.submit([initialSparseScene.finish()]);
    this.octreeProjection.finishInlineRebuild();
  }

  get volumeTexture() { return this.volumeA; }
  get rigidRenderBuffer() { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer() { return this.rigidSystem.motionBuffer; }
  setSelectedRigidBody(index: number) { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin,direction); }
  // Rendering contours the smooth resident level set when the quadtree
  // projection maintains one; the flux-form VOF field is near-binary and its
  // 0.5 contour is quantized to cell scale. Diagnostics keep reading the VOF
  // field through volumeTexture.
  private get adaptiveProjection() { return this.quadtreeProjection ?? this.octreeProjection; }
  get surfaceFieldTexture() { return this.adaptiveProjection?.levelSetTexture ?? this.volumeA; }
  get sparseVoxelSceneSource() { return this.octreeProjection?.sparseVoxelSceneSource; }
  get sparseVoxelRenderSource() {
    const source = this.octreeProjection?.sparseVoxelRenderSource;
    if (this.octreeProjection) this.applyOctreeInfo(this.octreeProjection);
    return source;
  }
  get fluidBrickAtlasSamplingSource() { return this.octreeProjection?.fluidBrickAtlasSamplingSource; }
  get sparseSurfaceBand(): SparseSurfaceBandGPUSource | undefined { return this.octreeProjection?.sparseSurfaceBandSource; }
  get columnBaseTexture() { return this.heightA; }
  get gridCellTexture() { return this.adaptiveProjection?.topologyTexture; }
  get velocityTexture() { return this.velocityA; }
  get secondaryParticles() { return this.secondaryParticleSystem?.renderSource; }
  get gridPressureSamplesTexture() { return this.adaptiveProjection?.pressureSamplesTexture; }
  get gridPressureTexture() { return this.adaptiveProjection?.pressureTexture; }
  get gridDivergenceTexture() { return this.adaptiveProjection?.divergenceTexture; }
  ensureGridDiagnosticTextures() {
    if (!this.octreeProjection?.ensureDiagnosticTextures()) return;
    const encoder = this.device.createCommandEncoder({ label: "Initialize lazy octree diagnostic fields" });
    this.octreeProjection.encodeOverlayMaterialization(encoder);
    this.device.queue.submit([encoder.finish()]);
    this.applyOctreeInfo(this.octreeProjection);
  }
  /** Instrumentation view: velocity after advection/forces and before quadtree projection. */
  get preProjectionVelocityTexture() { return this.velocityB; }

  private initializeVolume() {
    const { nx, ny, nz } = this.info, c = this.scene.container;
    const data = new Float32Array(nx * ny * nz), dam = damBreakFractions(c.fillFraction);
    const terrainHeights = terrainColumnHeights(this.scene, nx, nz), cellHeight = c.height_m / ny;
    let initialSum = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const aboveGround = (j + 0.5) * cellHeight > terrainHeights[i + nx * k];
      const brickWet = initialFluidBrickContainsCell(this.scene, i, j, k, [nx, ny, nz]);
      const fill = aboveGround && combineInitialBrickWet(this.scene, brickWet, this.scene.fluid.initialCondition === "dam-break"
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

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture, heightIn: GPUTexture, heightOut: GPUTexture, predictedVelocity: GPUTexture = velocityIn, reversedVelocity: GPUTexture = velocityIn, transport: GPUTexture = this.transportA, surfaceIn: GPUTexture = this.adaptiveProjection?.levelSetTexture ?? volumeIn, sampleBulkAtlas = false, sparseTargetKind: "none" | "velocity" | "occupancy" = "none") {
    const candidate = this.octreeProjection?.fluidBrickAtlasSamplingSource;
    const source: FluidBrickAtlasSamplingSource | undefined = candidate?.mode === "mirror" ? candidate : undefined;
    const atlas = sampleBulkAtlas ? source : undefined;
    const sparseRequested = sparseTargetKind === "velocity" ? this.sparseVelocityAdvectionRequested
      : sparseTargetKind === "occupancy" ? this.sparseOccupancyFluxPreparationRequested : false;
    const sparse = sparseRequested && source?.sparseDispatchSafe ? source : undefined;
    const control = atlas && sparse ? this.bulkAtlasControlSampleSparse : atlas ? this.bulkAtlasControlSampleDense : sparse ? this.bulkAtlasControlSparse : this.bulkAtlasControlOff;
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
      { binding: 21, resource: this.terrainTexture.createView() },
      { binding: 22, resource: { buffer: atlas?.pageTable ?? this.bulkAtlasFallbackPageTable } },
      { binding: 23, resource: atlas?.velocity ?? this.bulkAtlasFallbackVelocity.createView() },
      { binding: 24, resource: { buffer: (atlas ?? sparse)?.params ?? this.bulkAtlasFallbackParams } },
      { binding: 25, resource: { buffer: control } },
      { binding: 26, resource: { buffer: sparse?.bulkWorklist ?? this.bulkAtlasFallbackWorklist } },
      { binding: 27, resource: { buffer: this.occupancyColumns } }
    ] });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup, sparseBulkTargets = false) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    const source = this.octreeProjection?.fluidBrickAtlasSamplingSource;
    if (sparseBulkTargets && this.sparseVelocityAdvectionRequested && source?.mode === "mirror" && source.sparseDispatchSafe) {
      pass.dispatchWorkgroupsIndirect(source.bulkWorklist, FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES);
    } else pass.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
  }
  private dispatchTransport(pass: GPUComputePassEncoder, group: GPUBindGroup, sparseTransportTargets: boolean, paddedWorkgroups: [number, number, number]) {
    pass.setPipeline(this.buildTransportPipeline); pass.setBindGroup(0, group);
    const source = this.octreeProjection?.fluidBrickAtlasSamplingSource;
    if (sparseTransportTargets && this.sparseTransportPreparationRequested && source?.mode === "mirror" && source.sparseDispatchSafe) {
      pass.dispatchWorkgroupsIndirect(source.bulkWorklist, FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES);
    } else pass.dispatchWorkgroups(...paddedWorkgroups);
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
      fluidBrickCapacity: projection.fluidBrickCapacity,
      quadtreeLeafCount: octree.leafCount,
      quadtreePressureSampleCount: octree.pressureSampleCount,
      quadtreeLiquidDofCount: octree.liquidDofCount,
      quadtreeMaximumNeighborRatio: octree.maximumNeighborRatio,
      quadtreeMaximumFluidScale: octree.maximumFluidScale,
      quadtreePressureIterationsUsed: octree.pressureIterationsUsed,
      quadtreePressureIterationBudget: octree.pressureIterationBudget,
      quadtreePressureIterationHardBudget: octree.pressureIterationHardBudget,
      pressureRowCapacity: octree.pressureRowCapacity,
      pressureEntryCapacity: octree.pressureEntryCapacity,
      pressureRequiredRows: octree.pressureRequiredRows,
      pressureRequiredEntries: octree.pressureRequiredEntries,
      pressureCapacityOverflow: octree.pressureCapacityOverflow,
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
    const coarseHMin = Math.min(c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz);
    // Sparse phi advection is semi-Lagrangian and does not force the global
    // pressure solve onto the fine geometric timestep. Preserve the coarse
    // Chebyshev cadence unless explicit fine dynamics is enabled.
    const surfaceFactor = this.octreeProjection?.requiresFineSurfaceTimestep ? this.octreeProjection.sparseSurfaceRefinementFactor : 1;
    const hMin = coarseHMin / surfaceFactor;
    const inflowSpeed = this.scene.fluid.inflow ? Math.hypot(this.scene.fluid.inflow.velocity_m_s.x, this.scene.fluid.inflow.velocity_m_s.y, this.scene.fluid.inflow.velocity_m_s.z) : 0;
    const substeps = this.adaptiveProjection ? proactiveQuadtreeSubsteps(
      this.info.maxSpeed_m_s ?? 0,
      inflowSpeed,
      Math.hypot(this.scene.fluid.gravity_m_s2.x, this.scene.fluid.gravity_m_s2.y, this.scene.fluid.gravity_m_s2.z),
      delta,
      hMin,
      64,
      rho,
      sigma
    ) : 1;
    const dt = delta / substeps; this.info.lastDt_s = dt; this.info.lastSubsteps = substeps;
    const activeBodies = bodies.slice(0, 12);
    this.rigidSystem.syncBodies(activeBodies); this.info.encodedSteps = (this.info.encodedSteps ?? 0) + substeps;
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
      inlineRebuildEncoded = this.octreeProjection.encodeInlineRebuild(encoder, timing && this.querySet ? {
        querySet: this.querySet,
        beginningOfPassWriteIndex: timing.start,
        endOfPassWriteIndex: timing.end,
      } : undefined);
    }
    for (let substep = 0; substep < substeps; substep += 1) {
      // The first rebuild was encoded above so topology is ready before any
      // dynamics. If CFL control subdivides this advance, phi moves after each
      // projection; rebuild again before the next substep so a newly exposed
      // interface can never remain inside a coarse pressure leaf.
      if (substep > 0 && this.octreeProjection) this.octreeProjection.encodeInlineRebuild(encoder);
      const bulkSource = this.octreeProjection?.fluidBrickAtlasSamplingSource;
      if (bulkSource?.mode === "mirror") {
        bulkSource.encodeBulkRefresh(encoder, this.octreeProjection!.levelSetTexture, this.velocityA, dt);
      }
      const sparseBulkTargets = this.sparseVelocityAdvectionRequested
        && bulkSource?.mode === "mirror" && bulkSource.sparseDispatchSafe;
      const sparseTransportTargets = this.sparseTransportPreparationRequested
        && bulkSource?.mode === "mirror" && bulkSource.sparseDispatchSafe;
      const sparseOccupancyFluxTargets = this.sparseOccupancyFluxPreparationRequested
        && bulkSource?.mode === "mirror" && bulkSource.sparseDispatchSafe;
      if (sparseOccupancyFluxTargets) encoder.copyBufferToBuffer(
        bulkSource.bulkWorklist, FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES,
        this.occupancyCell64Dispatch, 0, 12,
      );
      const sparseExtrapolationTargets = this.octreeProjection?.usesSparseVelocityExtrapolation ?? false;
      if (bulkSource && (sparseBulkTargets || sparseTransportTargets || sparseOccupancyFluxTargets || sparseExtrapolationTargets)) {
        const clear = encoder.beginComputePass({ label: "Clear retired sparse compatibility payloads" });
        if (sparseBulkTargets || sparseExtrapolationTargets) {
          clear.setPipeline(this.retiredVelocityClearPipeline);
          for (const group of this.retiredVelocityClearGroups) {
            clear.setBindGroup(0, group);
            clear.dispatchWorkgroupsIndirect(bulkSource.bulkWorklist, FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES);
          }
        }
        if (sparseTransportTargets) {
          clear.setPipeline(this.retiredTransportClearPipeline);
          for (const group of this.retiredTransportClearGroups) {
            clear.setBindGroup(0, group);
            clear.dispatchWorkgroupsIndirect(bulkSource.bulkWorklist, FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES);
          }
        }
        if (sparseOccupancyFluxTargets && this.transportConservativeVolume) {
          clear.setPipeline(this.retiredFluxScaleClearPipeline);
          clear.setBindGroup(0, this.retiredFluxScaleClearGroup);
          clear.dispatchWorkgroupsIndirect(bulkSource.bulkWorklist, FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES);
        }
        clear.end();
      }
      {
        const timing = this.timing("advection_ms");
        const paddedWorkgroups: [number, number, number] = [Math.ceil((this.info.nx + 2) / 4), Math.ceil((this.info.ny + 2) / 4), Math.ceil((this.info.nz + 2) / 4)];
        // Occupancy, transport extrapolation, and flux scales only read the
        // projected state, so they share one pass ahead of the predictor.
        if (sparseOccupancyFluxTargets) encoder.clearBuffer(this.occupancyColumns);
        const prep = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start } } : undefined);
        prep.setBindGroup(0, this.occupancyGroup);
        if (sparseOccupancyFluxTargets) {
          // Fail closed entirely on-GPU: generation zero / an empty first-step
          // list executes the dense column scan, while populated frames return
          // immediately from this area-only sentinel dispatch.
          prep.setPipeline(this.buildOccupancyPipeline);
          prep.dispatchWorkgroups(Math.ceil(this.info.nx / 8), Math.ceil(this.info.nz / 8), 1);
          prep.setPipeline(this.buildSparseOccupancyPipeline);
          prep.dispatchWorkgroupsIndirect(this.occupancyCell64Dispatch, 0);
          prep.setPipeline(this.resolveSparseOccupancyPipeline);
          prep.dispatchWorkgroups(Math.ceil(this.info.nx / 8), Math.ceil(this.info.nz / 8), 1);
        } else {
          prep.setPipeline(this.buildOccupancyPipeline);
          prep.dispatchWorkgroups(Math.ceil(this.info.nx / 8), Math.ceil(this.info.nz / 8), 1);
        }
        this.dispatchTransport(prep, this.transportFromCurrentGroup, sparseTransportTargets, paddedWorkgroups);
        if (this.transportConservativeVolume) {
          prep.setPipeline(this.buildFluxScalesPipeline);
          prep.setBindGroup(0, this.fluxScaleGroup);
          if (sparseOccupancyFluxTargets) prep.dispatchWorkgroupsIndirect(this.occupancyCell64Dispatch, 0);
          else prep.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
        }
        prep.end();
        const predict = encoder.beginComputePass(timing && this.querySet && this.velocityTransport === "semi-lagrangian" ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.end } } : undefined);
        this.dispatch(predict, this.advectPipeline, this.advectGroup, sparseBulkTargets); predict.end();
        if (this.velocityTransport === "maccormack" && this.transportFromPredictedGroup) {
          const predictedTransport = encoder.beginComputePass();
          this.dispatchTransport(predictedTransport, this.transportFromPredictedGroup, sparseTransportTargets, paddedWorkgroups); predictedTransport.end();
          const reverse = encoder.beginComputePass(); this.dispatch(reverse, this.reversePipeline, this.reverseGroup, sparseBulkTargets); reverse.end();
          const correct = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(correct, this.correctPipeline, this.correctGroup, sparseBulkTargets); correct.end();
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
              materialization: this.octreeProjection.hasDiagnosticTextures ? timestampWrites(this.timing("materialization_ms")) : undefined
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
    if (activeBodies.length > 0) {
      this.quadtreeProjection?.encodeBodyImpulseExchange(encoder, this.rigidExchangeBuffer);
      const cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz);
      this.rigidSystem.encode(encoder, delta, cellVolume, substeps, c.height_m / this.info.ny);
    }
    // Publish the final substep's resident fields into the shared sparse-brick
    // world. The topology and payload stay GPU-resident; rendering consumes
    // compact debug records and subsequent voxel kernels consume the same ABI.
    if (this.octreeProjection) {
      const timestampWrites = (timing: { start: number; end: number } | undefined) => timing && this.querySet ? {
        querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end
      } : undefined;
          this.octreeProjection.encodeSparseBrickWorld(encoder, {
            residency: timestampWrites(this.timing("fluidResidency_ms")),
            publication: timestampWrites(this.timing("sparsePublication_ms"))
          }, dt, this.octreeProjection.fluidBrickAtlasSamplingSource?.mode === "mirror");
    }
    encoder.clearBuffer(this.reductionBuffer); { const timing = this.timing("diagnostics_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.reductionPipeline, this.reductionGroup); pass.end(); }
    if (totalTiming && this.querySet) { const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: totalTiming.end } }); pass.end(); }
    if (this.querySet && this.queryResolve && this.queryCount > 0) encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.queryResolve, 0);
    const submittedAt = performance.now(); this.device.queue.submit([encoder.finish()]);
    // Solver residuals are sampled only through the opt-in telemetry path.
    // Physics submission itself must not initiate a GPU-to-CPU map.
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
      this.quadtreeLastBodies = activeBodies.filter((body) => body.description.motion === "static");
      if (this.quadtreeRebuildRetrySteps > 0) this.quadtreeRebuildRetrySteps -= 1;
      if (!inlineRebuildEncoded && this.quadtreeRebuildRetrySteps === 0 && !this.quadtreeRebuildPending && !this.quadtreeReadyProjection && this.shouldKickQuadtreeRebuild()) this.kickQuadtreeRebuild();
    }
    if (!this.wallTimingPending) { this.wallTimingPending = true; void this.device.queue.onSubmittedWorkDone().then(() => { this.info.gpuQueueWall_ms = performance.now() - submittedAt; this.info.gpuQueueSimulation_s = delta; }).catch(() => { /* Device loss is handled by the renderer. */ }).finally(() => { this.wallTimingPending = false; }); }
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
      const [, , surfaceDiagnostics, fluidBrickStats, fluidBulkBrickStats, sparseSurfaceStats, fluidBrickAtlasStats] = await Promise.all([
        mapPromise, quadtreeDiagnostics, surfaceDiagnosticsPromise, this.octreeProjection?.readFluidBrickResidencyStats(), this.octreeProjection?.readFluidBulkBrickResidencyStats(), this.octreeProjection?.readSparseSurfaceBandStats(), this.octreeProjection?.readFluidBrickAtlasStats(),
      ]);
    if(fluidBrickStats){this.info.fluidBrickCapacity=fluidBrickStats.capacity;this.info.fluidBrickResidentCount=fluidBrickStats.resident;this.info.fluidBrickCoreCount=fluidBrickStats.core;this.info.fluidBrickHaloCount=fluidBrickStats.halo;this.info.fluidBrickActivatedCount=fluidBrickStats.activated;this.info.fluidBrickRetiredCount=fluidBrickStats.retired;this.info.fluidBrickGeneration=fluidBrickStats.generation;}
    if(fluidBulkBrickStats){this.info.fluidBulkBrickResidentCount=fluidBulkBrickStats.resident;this.info.fluidBulkBrickHaloCount=fluidBulkBrickStats.halo;this.info.fluidBulkBrickActivatedCount=fluidBulkBrickStats.activated;this.info.fluidBulkBrickRetiredCount=fluidBulkBrickStats.retired;}
    if(fluidBrickAtlasStats){this.info.fluidBrickAtlasCapacity=fluidBrickAtlasStats.capacity;this.info.fluidBrickAtlasResidentTiles=fluidBrickAtlasStats.residentTiles;this.info.fluidBrickAtlasOverflow=fluidBrickAtlasStats.overflow;this.info.fluidBrickAtlasMaxPhiError=fluidBrickAtlasStats.maxAbsPhiError;this.info.fluidBrickAtlasMaxVelocityError=fluidBrickAtlasStats.maxAbsVelocityError;this.info.fluidBrickAtlasMaxPhiErrorManual=fluidBrickAtlasStats.maxAbsPhiErrorManual;this.info.fluidBrickAtlasMaxVelocityErrorManual=fluidBrickAtlasStats.maxAbsVelocityErrorManual;}
    if(sparseSurfaceStats){this.info.sparseSurfaceLogicalPages=sparseSurfaceStats.logicalPageCount;this.info.sparseSurfacePageCapacity=sparseSurfaceStats.physicalPageCapacity;this.info.sparseSurfaceResidentPages=sparseSurfaceStats.resident;this.info.sparseSurfaceCorePages=sparseSurfaceStats.core;this.info.sparseSurfaceHaloPages=sparseSurfaceStats.halo;this.info.sparseSurfaceActivatedPages=sparseSurfaceStats.activated;this.info.sparseSurfaceRetiredPages=sparseSurfaceStats.retired;this.info.sparseSurfaceOverflow=sparseSurfaceStats.overflow;this.info.sparseSurfacePeakPages=sparseSurfaceStats.peakResident;}
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
      this.info.activeSampleCount = this.octreeProjection.info.liquidDofCount;
      this.info.activeCompressionRatio = this.octreeProjection.info.compressionRatio;
      this.info.compressionRatio = this.octreeProjection.info.compressionRatio;
      this.info.quadtreePressureSampleCount = this.octreeProjection.info.pressureSampleCount;
      this.info.quadtreeLiquidDofCount = this.octreeProjection.info.liquidDofCount;
      this.info.quadtreePressureIterationsUsed = this.octreeProjection.info.pressureIterationsUsed;
      this.info.quadtreePressureIterationBudget = this.octreeProjection.info.pressureIterationBudget;
          this.info.quadtreePressureIterationHardBudget = this.octreeProjection.info.pressureIterationHardBudget;
          this.info.pressureRowCapacity = this.octreeProjection.info.pressureRowCapacity;
          this.info.pressureEntryCapacity = this.octreeProjection.info.pressureEntryCapacity;
          this.info.pressureRequiredRows = this.octreeProjection.info.pressureRequiredRows;
          this.info.pressureRequiredEntries = this.octreeProjection.info.pressureRequiredEntries;
          this.info.pressureCapacityOverflow = this.octreeProjection.info.pressureCapacityOverflow;
          this.info.pressureSolver = this.octreeProjection.pressureSolverLabel;
    }
    if (queryBytes > 0) {
      const times = new BigUint64Array(buffer.getMappedRange(16, queryBytes));
      const stageByField: Partial<Record<GPUPhysicsTimingField, GPUPhysicsStageId>> = { preparation_ms: "preparation", layerConstruction_ms: "topology", advection_ms: "advection", conditioning_ms: "conditioning", remeshing_ms: "remeshing", pressure_ms: "pressure", projection_ms: "projection", extrapolation_ms: "extrapolation", materialization_ms: "materialization", surfaceUpdate_ms: "surfaceUpdate", rigidCoupling_ms: "rigidCoupling", spray_ms: "spray", fluidResidency_ms: "fluidResidency", sparsePublication_ms: "sparsePublication", diagnostics_ms: "diagnostics" };
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
    for (const texture of new Set([this.velocityA, this.velocityB, this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB, this.terrainTexture, this.transportA, this.transportB, this.fluxScales])) texture.destroy();
    this.params.destroy(); this.reductionBuffer.destroy(); this.sharpenBuffer.destroy(); this.occupancyColumns.destroy(); this.occupancyCell64Dispatch.destroy(); this.bulkAtlasFallbackPageTable.destroy(); this.bulkAtlasFallbackParams.destroy(); this.bulkAtlasFallbackVelocity.destroy(); this.bulkAtlasFallbackWorklist.destroy(); this.bulkAtlasControlSampleSparse.destroy(); this.bulkAtlasControlSampleDense.destroy(); this.bulkAtlasControlSparse.destroy(); this.bulkAtlasControlOff.destroy(); this.rigidSystem.destroy(); this.rigidExchangeBuffer.destroy(); this.statsReadbackBuffer?.destroy(); this.querySet?.destroy(); this.queryResolve?.destroy();
  }
}
