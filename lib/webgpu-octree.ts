import type { SceneDescription } from "./model";
import { combineInitialBrickWet, damBreakFractions, initialFluidBrickContainsCell } from "./initial-fluid";
import { signedDistanceFromVolume } from "./quadtree-tall-cell-grid";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import { WebGPUQuadtreeSurfaceState, type SurfaceInflowState } from "./webgpu-quadtree-builder";
import { OctreeSparseBrickWorld } from "./webgpu-octree-sparse-bricks";
import {
  FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_RETIRED_CANDIDATE_DISPATCH_OFFSET_BYTES,
  FLUID_TILE_RETIRED_DISPATCH_OFFSET_BYTES,
} from "./webgpu-fluid-brick-residency";
import {
  WebGPUSparseSurfaceBand,
  type SparseSurfaceBandGPUSource,
  type SparseSurfaceBandMode,
} from "./webgpu-sparse-surface-band";

export interface OctreeProjectionOptions {
  pressureIterations: number;
  maximumLeafSize?: 2 | 4 | 8 | 16 | 32;
  /** 0 = finest cells everywhere; 1 = full distance-graded coarsening. */
  adaptivity?: number;
  /** Pure-phase cells farther from liquid/solid interfaces than this finest-cell band may remain coarse. */
  interfaceRefinementBandCells?: number;
  /** Adds up to eight finest cells of refinement support around curved or strongly straining surface regions. */
  surfaceDetailStrength?: number;
  /** Dynamic fine level-set pages around phi=0; off preserves the dense-only path. */
  sparseSurfaceBand?: "off" | SparseSurfaceBandMode;
  /** Fine level-set samples per coarse transport cell edge. */
  surfaceRefinementFactor?: 1 | 2 | 4;
  /** Two-sided sparse phi support measured in fine cells. */
  sparseSurfaceBandCells?: number;
  /** Fraction of the logical fine-page lattice backed by physical slots. */
  sparseSurfacePageFraction?: number;
  /** Brick-pooled phi/velocity atlas mirrored from the dense fields. */
  brickAtlas?: boolean;
  /** Velocity-swept brick residency support plus downstream activation. */
  brickPreActivation?: boolean;
  jacobiRelaxation?: number;
  extrapolationSweeps?: number;
  /**
   * Leaf pressure solve strategy. "auto" (= "chebyshev") prefix-sum-compacts
   * the liquid leaf origins, assembles each row's diagonal / flux / merged
   * neighbor table once per solve, then applies a Chebyshev-accelerated
   * polynomial in row-parallel indirect dispatches. "compact" retains the
   * warm-started weighted-Jacobi ladder for A/B. "megakernel" instead
   * runs the whole iteration loop in one persistent single-workgroup dispatch
   * with a residual early exit — it wins only when solves converge within a
   * few iterations. "dense" is the legacy one-thread-per-finest-cell ladder
   * kept for A/B comparison.
   */
  leafSolver?: "auto" | "dense" | "compact" | "chebyshev" | "megakernel";
  /**
   * Start each compacted solve from the previous step's pressure field
   * (default) instead of clearing to zero. The legacy "dense" ladder always
   * cold-starts so it remains a faithful pre-compaction baseline.
   */
  pressureWarmStart?: boolean;
}

interface OctreeProjectionResources {
  velocityIn: GPUTexture;
  velocityOut: GPUTexture;
  velocityScratch: GPUTexture;
  rigidBodies: GPUBuffer;
  rigidExchange: GPUBuffer;
  terrain: GPUTexture;
}

function octreeLeafSize(value: number): 2 | 4 | 8 | 16 | 32 {
  const rounded = Math.max(2, Math.round(value));
  if (rounded >= 32) return 32;
  if (rounded >= 16) return 16;
  if (rounded >= 8) return 8;
  return rounded <= 2 ? 2 : 4;
}

/**
 * A GPU-resident, pressure-only octree projection.
 *
 * The owner array is deliberately dense: every finest cell stores the origin
 * and size of its octree leaf, while pressure exists only at leaf origins.
 * This trades a small, predictable amount of memory for a topology rebuild
 * that needs neither prefix sums nor a GPU -> CPU synchronization point.
 */
export class WebGPUOctreeProjection {
  readonly preconditioner = "jacobi" as const;
  readonly canEncodeInlineRebuild = true;
  readonly levelSetTexture: GPUTexture;
  readonly info: {
    leafCount: number;
    pressureSampleCount: number;
    liquidDofCount: number;
    faceCount: number;
    mlsProjectionRowCount: number;
    tallSegmentCount: number;
    ghostFaceCount: number;
    maximumNeighborRatio: number;
    maximumFluidScale: number;
    compressionRatio: number;
    allocatedBytes: number;
    pressureIterationsUsed: number;
    pressureIterationBudget: number;
    pressureIterationHardBudget: number;
    pressureConverged?: boolean;
    velocityClampCount: number;
    factorLevelCount: number;
    multigridLevelCount: number;
    multigridCoarsestDofs: number;
    pressurePhaseTimings?: Record<string, number>;
    topologyReadbackBytes: number;
    topologyReused: boolean;
    topologyReuseCount: number;
    gpuConstruction_ms: number;
    gpuConstructionKernel_ms: number;
    gpuSparsePack_ms: number;
    cpuTopologyPack_ms: number;
    cpuRedistance_ms: number;
    cpuQuadtreeDecode_ms: number;
    cpuTallGrid_ms: number;
    cpuVariationalAssembly_ms: number;
    cpuSystemPack_ms: number;
    cpuICFactorization_ms: number;
    cpuResourceUpload_ms: number;
  };
  levelSetMismatchFraction = 0;
  relativeResidual?: number;
  residualRms?: number;
  initialResidualRms?: number;

  private readonly topology: GPUBuffer;
  private readonly pressureA: GPUBuffer;
  private readonly pressureB: GPUBuffer;
  private readonly compaction: GPUBuffer;
  private readonly leafHeaders: GPUBuffer;
  private readonly leafEntries: GPUBuffer;
  private readonly leafFrontier: GPUBuffer;
  private readonly solveDispatch: GPUBuffer;
  private readonly topologyCandidateDispatch: GPUBuffer;
  private readonly solidCells: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly shader: GPUShaderModule;
  private readonly diagnosticLayout: GPUBindGroupLayout;
  private readonly diagnosticPipelineLayout: GPUPipelineLayout;
  private readonly diagnosticShader: GPUShaderModule;
  private readonly couplingLayout: GPUBindGroupLayout;
  private readonly couplingPipelineLayout: GPUPipelineLayout;
  private readonly couplingShader: GPUShaderModule;
  private readonly surfaceState: WebGPUQuadtreeSurfaceState;
  private readonly sparseSurfaceBand?: WebGPUSparseSurfaceBand;
  private readonly sparseBrickWorld: OctreeSparseBrickWorld;
  private readonly groups: { ab: GPUBindGroup; ba: GPUBindGroup; extrapolateOut: GPUBindGroup; extrapolateScratch: GPUBindGroup };
  private topologyDiagnosticTexture?: GPUTexture;
  private pressureSamplesDiagnosticTexture?: GPUTexture;
  private pressureDiagnosticTexture?: GPUTexture;
  private divergenceDiagnosticTexture?: GPUTexture;
  private diagnosticGroups?: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  private readonly couplingGroups: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  private rasterizeSolidsPipeline!: GPUComputePipeline;
  private resetPipeline!: GPUComputePipeline;
  private refinePipeline!: GPUComputePipeline;
  private balancePipeline!: GPUComputePipeline;
  private rasterizeSolidsActivePipeline!: GPUComputePipeline;
  private resetActivePipeline!: GPUComputePipeline;
  private refineActivePipeline!: GPUComputePipeline;
  private balanceActivePipeline!: GPUComputePipeline;
  private rasterizeSolidsRetiredPipeline!: GPUComputePipeline;
  private resetRetiredPipeline!: GPUComputePipeline;
  private refineRetiredPipeline!: GPUComputePipeline;
  private balanceRetiredPipeline!: GPUComputePipeline;
  private readonly refineLevelPipelines = new Map<number, {
    full: GPUComputePipeline;
    active: GPUComputePipeline;
    retired: GPUComputePipeline;
  }>();
  private readonly refineCoarsePipelines = new Map<number, GPUComputePipeline>();
  private readonly balanceCoarsePipelines = new Map<number, GPUComputePipeline>();
  private jacobiPipeline!: GPUComputePipeline;
  private planPipeline!: GPUComputePipeline;
  private scanPipeline!: GPUComputePipeline;
  private emitPipeline!: GPUComputePipeline;
  private beginFrontierPipeline!: GPUComputePipeline;
  private filterFrontierPipeline!: GPUComputePipeline;
  private appendFrontierPipeline!: GPUComputePipeline;
  private appendFrontierActivePipeline!: GPUComputePipeline;
  private appendFrontierRetiredPipeline!: GPUComputePipeline;
  private finalizeFrontierPipeline!: GPUComputePipeline;
  private assemblePipeline!: GPUComputePipeline;
  private assembleCoarsePipeline!: GPUComputePipeline;
  private gatherBodyCouplingPipeline!: GPUComputePipeline;
  private applyBodyCouplingPipeline!: GPUComputePipeline;
  private iteratePipeline!: GPUComputePipeline;
  private iterateChebyshevPipeline!: GPUComputePipeline;
  private solvePipeline!: GPUComputePipeline;
  private pressureImpulsePipeline!: GPUComputePipeline;
  private reconstructSmallGradientsPipeline!: GPUComputePipeline;
  private reconstructGradientsPipeline!: GPUComputePipeline;
  private projectPipeline!: GPUComputePipeline;
  private projectSmallLeavesPipeline!: GPUComputePipeline;
  private projectLeavesPipeline!: GPUComputePipeline;
  private extrapolateSeedPipeline!: GPUComputePipeline;
  private extrapolatePipeline!: GPUComputePipeline;
  private materializePipeline!: GPUComputePipeline;
  private readonly maxLeafSize: number;
  private readonly topologyTileSize: number;
  private readonly adaptivity: number;
  private readonly interfaceRefinementBandCells: number;
  private readonly surfaceDetailStrength: number;
  private readonly iterations: number;
  private readonly extrapolationSweeps: number;
  private readonly leafSolver: "dense" | "compact" | "chebyshev" | "megakernel";
  private readonly pressureWarmStart: boolean;
  private readonly workgroups: [number, number, number];
  private readonly linearBlocks: number;
  private couplingHasDynamicBodies = false;
  private topologyWorklistReady = false;
  private latestPressureInA = true;

  constructor(
    private readonly device: GPUDevice,
    private readonly scene: SceneDescription,
    private readonly dims: { nx: number; ny: number; nz: number },
    private readonly resources: OctreeProjectionResources,
    options: OctreeProjectionOptions,
    deferPipelineCompilation = false
  ) {
    const count = dims.nx * dims.ny * dims.nz;
    this.maxLeafSize = octreeLeafSize(options.maximumLeafSize ?? 16);
    this.adaptivity = Math.max(0, Math.min(1, options.adaptivity ?? 1));
    this.interfaceRefinementBandCells = Math.max(0, Math.min(32, Math.round(options.interfaceRefinementBandCells ?? 4)));
    this.surfaceDetailStrength = Math.max(0, Math.min(1, options.surfaceDetailStrength ?? 0));
    this.iterations = Math.max(8, Math.min(400, Math.round(options.pressureIterations)));
    this.extrapolationSweeps = Math.max(0, Math.min(8, Math.round(options.extrapolationSweeps ?? 4)));
    this.linearBlocks = Math.ceil(count / 256);
    // The row-parallel Chebyshev path is the default: it retains occupancy but
    // replaces four dispatch-bound Jacobi sweeps with one polynomial pass.
    // The single-workgroup megakernel is kept for very small/calm A/B cases.
    const requested = options.leafSolver ?? "auto";
    this.leafSolver = requested === "auto" ? "chebyshev" : requested;
    this.pressureWarmStart = options.pressureWarmStart ?? true;
    const cell = {
      x: scene.container.width_m / dims.nx,
      y: scene.container.height_m / dims.ny,
      z: scene.container.depth_m / dims.nz
    };
    // The surface-volume controller reads the same freshly rasterized VOS
    // field as pressure projection. Allocate it before the surface state so
    // open-fluid volume alpha*(1-s) and displaced volume alpha*s are exact
    // complements on every cell.
    this.solidCells = device.createBuffer({ label: "Octree VOS solid fractions and owners", size: Math.max(8, count * 8), usage: GPUBufferUsage.STORAGE });
    this.surfaceState = new WebGPUQuadtreeSurfaceState(
      device, dims, cell, resources.velocityOut, initialOctreeLevelSet(scene, dims, cell), undefined,
      undefined, false, false, true, true, this.solidCells
    );
    const sparseSurfaceMode = options.sparseSurfaceBand ?? "off";
    if (sparseSurfaceMode !== "off") {
      this.sparseSurfaceBand = new WebGPUSparseSurfaceBand(
        device,
        [dims.nx, dims.ny, dims.nz],
        [cell.x, cell.y, cell.z],
        this.surfaceState.texture,
        resources.velocityOut,
        this.solidCells,
        {
          mode: sparseSurfaceMode,
          refinementFactor: options.surfaceRefinementFactor ?? 2,
          bandCells: options.sparseSurfaceBandCells ?? 4,
          stencilCells: 5,
          retireAfterFrames: 3,
          maximumResidentFraction: options.sparseSurfacePageFraction ?? 0.75,
        },
      );
    }
    // Brick residency covers only sizing/detail support. The topology scheduler
    // dilates those tiles by one tile on-GPU: the complete 2:1 grading chain
    // travels 2+4+...+maxLeaf/2 < maxLeaf cells, so one neighboring tile is
    // sufficient without retaining maxLeaf-1 extra cells in the brick atlas.
    const topologyHaloCells = this.interfaceRefinementBandCells
      + 8 * this.surfaceDetailStrength;
    // Topology work is tile-atomic: max(brick, maximumLeaf) cells per axis,
    // so leaf-in-tile holds by construction for every legal size pair and the
    // rebuild never needs a full-domain fallback after initialization.
    this.topologyTileSize = Math.max(8, this.maxLeafSize);
    this.sparseBrickWorld = new OctreeSparseBrickWorld(device, scene, [dims.nx, dims.ny, dims.nz], {
      brickSize: 8,
      haloCells: topologyHaloCells,
      brickAtlas: options.brickAtlas ?? true,
      brickPreActivation: options.brickPreActivation ?? true,
      topologyTileBricks: this.topologyTileSize / 8
    });
    this.levelSetTexture = this.surfaceState.texture;
    // COPY_SRC on the owner map and pressure iterates exists solely for test
    // readbacks (leaf-size census, 2:1 balance, and finiteness audits); the
    // simulation itself never copies them out.
    this.topology = device.createBuffer({ label: "Octree dense owner map", size: Math.max(8, count * 8), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.pressureA = device.createBuffer({ label: "Octree leaf pressure A", size: Math.max(4, count * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.pressureB = device.createBuffer({ label: "Octree leaf pressure B", size: Math.max(4, count * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const compactBuffers = this.leafSolver !== "dense";
    // The scan totals are dead after leaf emission. The tail then doubles as
    // twelve resident rank-six generalized body-response vectors, avoiding a
    // ninth storage binding on minimum-limit WebGPU devices.
    this.compaction = device.createBuffer({
      label: "Octree leaf compaction, body coupling, and resident topology worklist",
      size: Math.max(60, 4 * (15 + 3 * this.linearBlocks + 12 * 8 + 2 * Math.ceil(count / (8 ** 3))), this.sparseBrickWorld.residency.tileWorklistByteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.leafHeaders = device.createBuffer({ label: "Octree leaf row headers", size: compactBuffers ? Math.max(32, count * 32) : 32, usage: GPUBufferUsage.STORAGE });
    this.leafEntries = device.createBuffer({ label: "Octree leaf matrix entries", size: compactBuffers ? Math.max(8, count * 48) : 8, usage: GPUBufferUsage.STORAGE });
    // Header + ping-pong leaf-origin lists + dense alive words. The lists are
    // evolved entirely on the GPU from active/retired topology tiles.
    this.leafFrontier = device.createBuffer({ label: "Persistent octree leaf frontier", size: Math.max(16, (4 + 3 * count) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.solveDispatch = device.createBuffer({ label: "Octree leaf solve and retired-topology dispatch", size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.topologyCandidateDispatch = device.createBuffer({ label: "Octree topology candidate-origin dispatch", size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    this.params = device.createBuffer({ label: "Octree projection parameters", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      // Both pressure buffers are writable so the persistent megakernel can
      // ping-pong iterates inside a single dispatch.
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } }
      ,{ binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.shader = device.createShaderModule({ label: "GPU-resident octree projection", code: octreeProjectionShader });
    void this.shader.getCompilationInfo().then((result) => {
      for (const message of result.messages) if (message.type === "error") console.error(`Octree GPU WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    }).catch(() => { /* device loss is handled by the owning solver */ });
    const group = (velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUBuffer, pressureOut: GPUBuffer) => device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: velocityIn.createView() },
      { binding: 1, resource: velocityOut.createView() },
      { binding: 2, resource: { buffer: this.compaction } },
      { binding: 3, resource: { buffer: this.topology } },
      { binding: 4, resource: { buffer: pressureIn } },
      { binding: 5, resource: { buffer: pressureOut } },
      { binding: 6, resource: { buffer: this.params } },
      { binding: 7, resource: this.levelSetTexture.createView() },
      { binding: 8, resource: { buffer: this.leafHeaders } },
      { binding: 9, resource: { buffer: this.leafEntries } },
      { binding: 10, resource: { buffer: resources.rigidBodies } },
      { binding: 11, resource: { buffer: this.solidCells } },
      { binding: 12, resource: resources.terrain.createView() }
      ,{ binding: 13, resource: { buffer: this.leafFrontier } }
    ] });
    this.groups = {
      ab: group(resources.velocityIn, resources.velocityOut, this.pressureA, this.pressureB),
      ba: group(resources.velocityIn, resources.velocityOut, this.pressureB, this.pressureA),
      extrapolateOut: group(resources.velocityOut, resources.velocityScratch, this.pressureA, this.pressureB),
      extrapolateScratch: group(resources.velocityScratch, resources.velocityOut, this.pressureA, this.pressureB)
    };
    this.diagnosticLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32uint", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
    ] });
    this.diagnosticPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.diagnosticLayout] });
    this.diagnosticShader = device.createShaderModule({ label: "GPU octree overlay materialization", code: octreeDiagnosticShader });
    this.couplingLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
    ] });
    this.couplingPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.couplingLayout] });
    this.couplingShader = device.createShaderModule({ label: "GPU octree pressure-to-body coupling", code: octreePressureCouplingShader });
    void this.couplingShader.getCompilationInfo().then((result) => {
      for (const message of result.messages) if (message.type === "error") console.error(`Octree coupling WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    }).catch(() => { /* device loss is handled by the owning solver */ });
    const couplingGroup = (pressure: GPUBuffer) => device.createBindGroup({ layout: this.couplingLayout, entries: [
      { binding: 0, resource: { buffer: pressure } },
      { binding: 1, resource: { buffer: this.topology } },
      { binding: 2, resource: { buffer: this.solidCells } },
      { binding: 3, resource: { buffer: resources.rigidBodies } },
      { binding: 4, resource: { buffer: resources.rigidExchange } },
      { binding: 5, resource: { buffer: this.params } },
      { binding: 6, resource: this.levelSetTexture.createView() }
    ] });
    this.couplingGroups = { pressureA: couplingGroup(this.pressureA), pressureB: couplingGroup(this.pressureB) };
    this.workgroups = [Math.ceil(dims.nx / 4), Math.ceil(dims.ny / 4), Math.ceil(dims.nz / 4)];
    const fullyCoarseEstimate = Math.ceil(count / (this.maxLeafSize ** 3));
    const approximateLeaves = Math.ceil(count * (1 - this.adaptivity) + fullyCoarseEstimate * this.adaptivity);
    const initialSolvePasses = this.leafSolver === "chebyshev" ? Math.ceil(this.iterations / 4) : this.iterations;
    this.info = {
      leafCount: approximateLeaves, pressureSampleCount: approximateLeaves, liquidDofCount: approximateLeaves,
      faceCount: 0, mlsProjectionRowCount: 0, tallSegmentCount: 0, ghostFaceCount: 0,
      maximumNeighborRatio: 2, maximumFluidScale: this.maxLeafSize, compressionRatio: approximateLeaves / Math.max(1, count),
      allocatedBytes: count * (this.leafSolver === "dense" ? 56 : 148) + this.compaction.size + 160
        + this.sparseBrickWorld.allocatedBytes + (this.sparseSurfaceBand?.allocatedBytes ?? 0),
      pressureIterationsUsed: initialSolvePasses, pressureIterationBudget: initialSolvePasses,
      pressureIterationHardBudget: initialSolvePasses, pressureConverged: undefined, velocityClampCount: 0,
      factorLevelCount: 0, multigridLevelCount: 0, multigridCoarsestDofs: 0, topologyReadbackBytes: 0,
      topologyReused: false, topologyReuseCount: 0, gpuConstruction_ms: 0, gpuConstructionKernel_ms: 0,
      gpuSparsePack_ms: 0, cpuTopologyPack_ms: 0, cpuRedistance_ms: 0, cpuQuadtreeDecode_ms: 0,
      cpuTallGrid_ms: 0, cpuVariationalAssembly_ms: 0, cpuSystemPack_ms: 0,
      cpuICFactorization_ms: 0, cpuResourceUpload_ms: 0
    };
    if (!deferPipelineCompilation) this.createPipelinesSync();
    this.writeParams(options.jacobiRelaxation ?? 0.8);
  }

  private descriptor(entryPoint: string): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shader, entryPoint } };
  }
  private refinementDescriptor(entryPoint: string, size: number): GPUComputePipelineDescriptor {
    return { layout: this.pipelineLayout, compute: { module: this.shader, entryPoint, constants: { targetRefinementSize: size } } };
  }
  private diagnosticDescriptor(): GPUComputePipelineDescriptor {
    return { layout: this.diagnosticPipelineLayout, compute: { module: this.diagnosticShader, entryPoint: "materializeOctreeFields" } };
  }
  private couplingDescriptor(): GPUComputePipelineDescriptor {
    return { layout: this.couplingPipelineLayout, compute: { module: this.couplingShader, entryPoint: "accumulatePressureImpulse" } };
  }

  private static readonly pipelineEntryPoints = [
    "rasterizeSolids", "resetTopology", "refineTopology", "balanceTopology", "jacobi",
    "rasterizeSolidsActive", "resetTopologyActive", "refineTopologyActive", "balanceTopologyActive",
    "rasterizeSolidsRetired", "resetTopologyRetired", "refineTopologyRetired", "balanceTopologyRetired",
    "beginFrontier", "filterFrontier", "appendFrontier", "appendFrontierActive", "appendFrontierRetired", "finalizeFrontier",
    "planLeaves", "scanLeafBlocks", "emitLeaves", "assembleSystem", "assembleCoarseSystem", "gatherBodyCoupling", "applyBodyCoupling", "iterateLeaves", "iterateChebyshev", "solveLeaves",
    "reconstructSmallGradients", "reconstructGradients", "project", "projectSmallLeaves", "projectLeaves", "extrapolateSeed", "extrapolate"
  ] as const;

  private assignPipelines(compiled: GPUComputePipeline[]) {
    [
      this.rasterizeSolidsPipeline, this.resetPipeline, this.refinePipeline, this.balancePipeline, this.jacobiPipeline,
      this.rasterizeSolidsActivePipeline, this.resetActivePipeline, this.refineActivePipeline, this.balanceActivePipeline,
      this.rasterizeSolidsRetiredPipeline, this.resetRetiredPipeline, this.refineRetiredPipeline, this.balanceRetiredPipeline,
      this.beginFrontierPipeline, this.filterFrontierPipeline, this.appendFrontierPipeline, this.appendFrontierActivePipeline, this.appendFrontierRetiredPipeline, this.finalizeFrontierPipeline,
      this.planPipeline, this.scanPipeline, this.emitPipeline, this.assemblePipeline, this.assembleCoarsePipeline, this.gatherBodyCouplingPipeline, this.applyBodyCouplingPipeline, this.iteratePipeline, this.iterateChebyshevPipeline, this.solvePipeline,
      this.reconstructSmallGradientsPipeline, this.reconstructGradientsPipeline, this.projectPipeline, this.projectSmallLeavesPipeline, this.projectLeavesPipeline, this.extrapolateSeedPipeline, this.extrapolatePipeline
    ] = compiled;
  }

  private createPipelinesSync() {
    this.assignPipelines(WebGPUOctreeProjection.pipelineEntryPoints.map((entryPoint) => this.device.createComputePipeline(this.descriptor(entryPoint))));
    for (let size = this.maxLeafSize; size >= 2; size >>= 1) this.refineLevelPipelines.set(size, {
      full: this.device.createComputePipeline(this.refinementDescriptor("refineTopology", size)),
      active: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyActive", size)),
      retired: this.device.createComputePipeline(this.refinementDescriptor("refineTopologyRetired", size)),
    });
    for (let size = this.maxLeafSize; size >= 16; size >>= 1) {
      this.refineCoarsePipelines.set(size, this.device.createComputePipeline(this.refinementDescriptor("refineTopologyCoarse", size)));
      this.balanceCoarsePipelines.set(size, this.device.createComputePipeline(this.refinementDescriptor("balanceTopologyCoarse", size)));
    }
    this.materializePipeline = this.device.createComputePipeline(this.diagnosticDescriptor());
    this.pressureImpulsePipeline = this.device.createComputePipeline(this.couplingDescriptor());
  }

  get topologyTexture() { return this.topologyDiagnosticTexture; }
  get pressureSamplesTexture() { return this.pressureSamplesDiagnosticTexture; }
  get pressureTexture() { return this.pressureDiagnosticTexture; }
  get divergenceTexture() { return this.divergenceDiagnosticTexture; }
  get hasDiagnosticTextures() { return this.diagnosticGroups !== undefined; }

  /** Allocate the dense scientific-overlay fields only after inspection asks for them. */
  ensureDiagnosticTextures(): boolean {
    if (this.diagnosticGroups) return false;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
    const size: GPUExtent3D = [this.dims.nx, this.dims.ny, this.dims.nz];
    this.topologyDiagnosticTexture = this.device.createTexture({ label: "Octree overlay topology", size, dimension: "3d", format: "rg32uint", usage });
    this.pressureSamplesDiagnosticTexture = this.device.createTexture({ label: "Octree overlay pressure ownership", size, dimension: "3d", format: "rgba32uint", usage });
    this.pressureDiagnosticTexture = this.device.createTexture({ label: "Octree mapped leaf pressure", size, dimension: "3d", format: "r32float", usage });
    this.divergenceDiagnosticTexture = this.device.createTexture({ label: "Octree projected divergence", size, dimension: "3d", format: "r32float", usage });
    const diagnosticGroup = (pressure: GPUBuffer, gradients: GPUBuffer) => this.device.createBindGroup({ layout: this.diagnosticLayout, entries: [
      { binding: 0, resource: { buffer: this.topology } },
      { binding: 1, resource: { buffer: pressure } },
      { binding: 2, resource: this.resources.velocityOut.createView() },
      { binding: 3, resource: this.levelSetTexture.createView() },
      { binding: 4, resource: this.topologyDiagnosticTexture!.createView() },
      { binding: 5, resource: this.pressureSamplesDiagnosticTexture!.createView() },
      { binding: 6, resource: this.pressureDiagnosticTexture!.createView() },
      { binding: 7, resource: this.divergenceDiagnosticTexture!.createView() },
      { binding: 8, resource: { buffer: this.params } },
      { binding: 9, resource: { buffer: gradients } },
      { binding: 10, resource: this.resources.velocityIn.createView() }
    ] });
    this.diagnosticGroups = {
      pressureA: diagnosticGroup(this.pressureA, this.pressureB),
      pressureB: diagnosticGroup(this.pressureB, this.pressureA)
    };
    this.info.allocatedBytes += this.dims.nx * this.dims.ny * this.dims.nz * 40;
    return true;
  }

  async initializePipelines(onProgress: (label: string, completed: number) => void) {
    const entries = WebGPUOctreeProjection.pipelineEntryPoints;
    const compiled: GPUComputePipeline[] = [];
    for (let i = 0; i < entries.length; i += 1) {
      onProgress(`Compile octree ${entries[i]}`, i);
      compiled.push(await this.device.createComputePipelineAsync(this.descriptor(entries[i])));
      onProgress(`Compile octree ${entries[i]}`, i + 1);
    }
    this.assignPipelines(compiled);
    for (let size = this.maxLeafSize; size >= 2; size >>= 1) this.refineLevelPipelines.set(size, {
      full: await this.device.createComputePipelineAsync(this.refinementDescriptor("refineTopology", size)),
      active: await this.device.createComputePipelineAsync(this.refinementDescriptor("refineTopologyActive", size)),
      retired: await this.device.createComputePipelineAsync(this.refinementDescriptor("refineTopologyRetired", size)),
    });
    for (let size = this.maxLeafSize; size >= 16; size >>= 1) {
      this.refineCoarsePipelines.set(size, await this.device.createComputePipelineAsync(this.refinementDescriptor("refineTopologyCoarse", size)));
      this.balanceCoarsePipelines.set(size, await this.device.createComputePipelineAsync(this.refinementDescriptor("balanceTopologyCoarse", size)));
    }
    onProgress("Compile octree overlay materialization", entries.length);
    this.materializePipeline = await this.device.createComputePipelineAsync(this.diagnosticDescriptor());
    onProgress("Compile octree overlay materialization", entries.length + 1);
    onProgress("Compile octree pressure-to-body coupling", entries.length + 1);
    this.pressureImpulsePipeline = await this.device.createComputePipelineAsync(this.couplingDescriptor());
    onProgress("Compile octree pressure-to-body coupling", entries.length + 2);
  }

  private writeParams(relaxation = 0.8) {
    const data = new ArrayBuffer(128);
    new Uint32Array(data, 0, 4).set([this.dims.nx, this.dims.ny, this.dims.nz, this.maxLeafSize]);
    new Float32Array(data, 16, 4).set([
      this.scene.container.width_m / this.dims.nx,
      this.scene.container.height_m / this.dims.ny,
      this.scene.container.depth_m / this.dims.nz,
      relaxation
    ]);
    new Uint32Array(data, 32, 4).set([Math.round(this.adaptivity * 1000), this.iterations, this.linearBlocks, 0]);
    // Megakernel residual tolerance, followed by the scaled Chebyshev spectrum.
    new Float32Array(data, 48, 4).set([1e-8, 0.01, 2.2, this.interfaceRefinementBandCells]);
    new Float32Array(data, 64, 4).set([
      this.scene.container.width_m,
      this.scene.container.height_m,
      this.scene.container.depth_m,
      sceneHasTerrain(this.scene) ? 1 : 0
    ]);
    const inflow = this.scene.fluid.inflow;
    const speed = inflow ? Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z) : 0;
    new Float32Array(data, 80, 4).set([inflow?.center_m.x ?? 0, inflow?.center_m.y ?? 0, inflow?.center_m.z ?? 0, inflow?.radius_m ?? 0]);
    new Float32Array(data, 96, 4).set([
      speed > 0 ? inflow!.velocity_m_s.x / speed : 0,
      speed > 0 ? inflow!.velocity_m_s.y / speed : 0,
      speed > 0 ? inflow!.velocity_m_s.z / speed : 0,
      inflow?.length_m ?? 0
    ]);
    new Float32Array(data, 112, 4).set([
      this.scene.fluid.density_kg_m3,
      this.scene.fluid.surfaceTension_N_m,
      this.scene.numerics.maxDt_s,
      this.surfaceDetailStrength
    ]);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  setCouplingBodies(count: number, hasDynamicBodies: boolean) {
    this.couplingHasDynamicBodies = hasDynamicBodies;
    this.device.queue.writeBuffer(this.params, 44, new Uint32Array([Math.max(0, Math.min(12, Math.floor(count)))]));
    this.device.queue.writeBuffer(this.params, 116, new Float32Array([hasDynamicBodies ? 1 : 0]));
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group = this.groups.ab) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(...this.workgroups);
  }

  encodeInlineRebuild(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) {
    // The first rebuild initializes every owner and solid cell. Thereafter the
    // previous publication's GPU-owned topology-tile list is the rebuild
    // domain: tiles span max(brick, maximumLeaf) cells, so every leaf lies
    // inside exactly one tile and partial rebuilds can never split a leaf.
    const active = this.topologyWorklistReady;
    if (active) {
      encoder.copyBufferToBuffer(
        this.sparseBrickWorld.residency.tileWorklist, 0,
        this.compaction, 0,
        this.sparseBrickWorld.residency.tileWorklistByteLength
      );
      // Dawn forbids one buffer being both writable storage and INDIRECT in a
      // pass; the dispatch args are staged into the dedicated indirect buffer.
      encoder.copyBufferToBuffer(
        this.sparseBrickWorld.residency.tileWorklist, FLUID_TILE_ACTIVE_DISPATCH_OFFSET_BYTES,
        this.solveDispatch, 0, 12
      );
      encoder.copyBufferToBuffer(
        this.sparseBrickWorld.residency.tileWorklist, FLUID_TILE_RETIRED_DISPATCH_OFFSET_BYTES,
        this.solveDispatch, 16, 12
      );
      encoder.copyBufferToBuffer(
        this.sparseBrickWorld.residency.tileWorklist, FLUID_TILE_ACTIVE_CANDIDATE_DISPATCH_OFFSET_BYTES,
        this.topologyCandidateDispatch, 0, 12
      );
      encoder.copyBufferToBuffer(
        this.sparseBrickWorld.residency.tileWorklist, FLUID_TILE_RETIRED_CANDIDATE_DISPATCH_OFFSET_BYTES,
        this.topologyCandidateDispatch, 16, 12
      );
    }
    const pass = encoder.beginComputePass({
      label: "Octree reset and refinement",
      ...(timestampWrites?.beginningOfPassWriteIndex !== undefined ? {
        timestampWrites: {
          querySet: timestampWrites.querySet,
          beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex,
        },
      } : {}),
    });
    const dispatch = (full: GPUComputePipeline, resident: GPUComputePipeline) => {
      pass.setPipeline(active ? resident : full);
      pass.setBindGroup(0, this.groups.ab);
      if (active) pass.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      else pass.dispatchWorkgroups(...this.workgroups);
    };
    const dispatchRetired = (pipeline: GPUComputePipeline) => {
      if (!active) return;
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.groups.ab);
      pass.dispatchWorkgroupsIndirect(this.solveDispatch, 16);
    };
    const candidateWorkgroups: [number, number, number] = [
      Math.ceil(this.dims.nx / 8), Math.ceil(this.dims.ny / 8), Math.ceil(this.dims.nz / 8)
    ];
    const dispatchCandidates = (full: GPUComputePipeline, resident: GPUComputePipeline) => {
      pass.setPipeline(active ? resident : full);
      pass.setBindGroup(0, this.groups.ab);
      if (active) pass.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      else pass.dispatchWorkgroups(...candidateWorkgroups);
    };
    const dispatchRetiredCandidates = (pipeline: GPUComputePipeline) => {
      if (!active) return;
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.groups.ab);
      pass.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 16);
    };
    dispatch(this.rasterizeSolidsPipeline, this.rasterizeSolidsActivePipeline);
    dispatchRetired(this.rasterizeSolidsRetiredPipeline);
    dispatch(this.resetPipeline, this.resetActivePipeline);
    dispatchRetired(this.resetRetiredPipeline);
    for (let size = this.maxLeafSize; size >= 2; size >>= 1) {
      if (size >= 16) {
        pass.setPipeline(this.refineCoarsePipelines.get(size)!);
        pass.setBindGroup(0, this.groups.ab);
        pass.dispatchWorkgroups(Math.ceil(this.dims.nx / size), Math.ceil(this.dims.ny / size), Math.ceil(this.dims.nz / size));
      } else {
        const level = this.refineLevelPipelines.get(size)!;
        dispatchCandidates(level.full, level.active);
        dispatchRetiredCandidates(level.retired);
      }
    }
    const balanceRounds = Math.max(1, Math.ceil(Math.log2(this.maxLeafSize)));
    for (let round = 0; round < balanceRounds; round += 1) {
      for (let size = this.maxLeafSize; size >= 16; size >>= 1) {
        pass.setPipeline(this.balanceCoarsePipelines.get(size)!);
        pass.setBindGroup(0, this.groups.ab);
        pass.dispatchWorkgroups(Math.ceil(this.dims.nx / size), Math.ceil(this.dims.ny / size), Math.ceil(this.dims.nz / size));
      }
      dispatchCandidates(this.balancePipeline, this.balanceActivePipeline);
      dispatchRetiredCandidates(this.balanceRetiredPipeline);
    }
    pass.end();

    // Evolve the persistent liquid-leaf frontier. Only cold initialization
    // walks the full finest lattice; subsequent frames filter the old compact
    // list and append replacement origins from the rebuilt topology tiles.
    const begin = encoder.beginComputePass({ label: "Begin persistent octree leaf frontier" });
    begin.setPipeline(this.beginFrontierPipeline); begin.setBindGroup(0, this.groups.ab); begin.dispatchWorkgroups(1);
    begin.end();
    encoder.copyBufferToBuffer(this.compaction, 48, this.topologyCandidateDispatch, 0, 12);
    const evolve = encoder.beginComputePass({
      label: "Evolve persistent octree leaf frontier",
      ...(timestampWrites?.endOfPassWriteIndex !== undefined ? {
        timestampWrites: {
          querySet: timestampWrites.querySet,
          endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex,
        },
      } : {}),
    });
    evolve.setBindGroup(0, this.groups.ab);
    if (active) {
      evolve.setPipeline(this.filterFrontierPipeline); evolve.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      evolve.setPipeline(this.appendFrontierActivePipeline); evolve.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      evolve.setPipeline(this.appendFrontierRetiredPipeline); evolve.dispatchWorkgroupsIndirect(this.solveDispatch, 16);
    } else {
      evolve.setPipeline(this.appendFrontierPipeline); evolve.dispatchWorkgroups(...this.workgroups);
    }
    evolve.setPipeline(this.finalizeFrontierPipeline); evolve.dispatchWorkgroups(1);
    evolve.end();
    // Reuse the candidate indirect buffer for pressure-row plan/emit; topology
    // candidate dispatches have already completed by this point.
    encoder.copyBufferToBuffer(this.compaction, 48, this.topologyCandidateDispatch, 0, 12);
    return true;
  }

  finishInlineRebuild() { this.info.topologyReuseCount += 1; }
  get extrapolationSweepCount() { return this.extrapolationSweeps; }
  get pressureSolverLabel() {
    if (this.leafSolver === "chebyshev") return `Octree Chebyshev-Jacobi · ${Math.ceil(this.iterations / 4)} parallel polynomial passes${this.couplingHasDynamicBodies ? " · frame-lagged rigid coupling" : ""}`;
    if (this.leafSolver === "megakernel" && !this.couplingHasDynamicBodies) return `Octree persistent Jacobi · up to ${this.iterations} sweeps`;
    return `Octree weighted Jacobi · ${this.iterations} fixed GPU sweeps`;
  }

  encode(
    encoder: GPUCommandEncoder,
    _nx: number,
    _ny: number,
    _nz: number,
    timestampWrites?: GPUComputePassTimestampWrites,
    detailedTimestampWrites?: {
      projection?: GPUComputePassTimestampWrites;
      extrapolation?: GPUComputePassTimestampWrites;
      materialization?: GPUComputePassTimestampWrites;
    }
  ) {
    const linear = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, blocks: number) => {
      pass.setPipeline(pipeline); pass.setBindGroup(0, this.groups.ab); pass.dispatchWorkgroups(blocks, 1, 1);
    };
    const gatherBodyCoupling = (pass: GPUComputePassEncoder, group: GPUBindGroup) => {
      if (!this.couplingHasDynamicBodies) return;
      pass.setPipeline(this.gatherBodyCouplingPipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(12, 1, 1);
    };
    const buildFrontierRows = (label: string, beginning?: GPUComputePassTimestampWrites) => {
      const compact = encoder.beginComputePass({ label, ...(beginning ? { timestampWrites: beginning } : {}) });
      compact.setPipeline(this.planPipeline); compact.setBindGroup(0, this.groups.ab); compact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      linear(compact, this.scanPipeline, 1);
      compact.setPipeline(this.emitPipeline); compact.setBindGroup(0, this.groups.ab); compact.dispatchWorkgroupsIndirect(this.topologyCandidateDispatch, 0);
      compact.end();
      encoder.copyBufferToBuffer(this.compaction, 8, this.solveDispatch, 0, 24);
    };
    // A monolithic rank-six response needs one global K^T p reduction per
    // pressure iterate. The optional single-workgroup megakernel cannot launch
    // those reductions. Chebyshev instead treats the uploaded rigid velocity
    // as prescribed for this solve and returns the new pressure impulse for the
    // next presentation batch: a deliberately frame-lagged partitioned split.
    const useMegakernel = this.leafSolver === "megakernel" && !this.couplingHasDynamicBodies;
    const useChebyshev = this.leafSolver === "chebyshev";
    const useLaggedRigidCoupling = useChebyshev && this.couplingHasDynamicBodies;
    const solvePasses = useChebyshev ? Math.ceil(this.iterations / 4) : this.iterations;
    this.info.pressureIterationsUsed = solvePasses;
    this.info.pressureIterationBudget = solvePasses;
    this.info.pressureIterationHardBudget = solvePasses;
    let pressureBufferSwaps = this.iterations;
    if (this.leafSolver === "dense") {
      encoder.clearBuffer(this.pressureA); encoder.clearBuffer(this.pressureB);
      const pressure = encoder.beginComputePass({ label: "Octree leaf Jacobi solve", ...(timestampWrites ? { timestampWrites } : {}) });
      for (let iteration = 0; iteration < this.iterations; iteration += 1) {
        const group = iteration % 2 === 0 ? this.groups.ab : this.groups.ba;
        gatherBodyCoupling(pressure, group);
        this.dispatch(pressure, this.jacobiPipeline, group);
      }
      pressure.end();
      // Dense Jacobi remains a validation baseline, but downstream affine
      // reconstruction/projection still consumes the persistent leaf rows.
      buildFrontierRows("Octree validation frontier rows");
    } else {
      // Compact the liquid leaf origins with a prefix-sum scan, assemble each
      // row's cached diagonal / flux / merged neighbor table once, then solve.
      // With warm start on, pressure is intentionally not cleared: the
      // persistent buffer seeds the solve with the previous converged field.
      if (!this.pressureWarmStart) { encoder.clearBuffer(this.pressureA); encoder.clearBuffer(this.pressureB); }
      buildFrontierRows("Octree leaf compaction", timestampWrites?.beginningOfPassWriteIndex !== undefined ? { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } : undefined);
      const pressure = encoder.beginComputePass({ label: "Octree leaf pressure solve", ...(timestampWrites?.endOfPassWriteIndex !== undefined ? { timestampWrites: { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } } : {}) });
      pressure.setPipeline(this.assemblePipeline);
      pressure.setBindGroup(0, this.groups.ab);
      pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      pressure.setPipeline(this.assembleCoarsePipeline);
      pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 12);
      if (useMegakernel) {
        linear(pressure, this.solvePipeline, 1);
        pressureBufferSwaps = 0;
      } else if (useChebyshev) {
        const acceleratedIterations = Math.ceil(this.iterations / 4);
        pressureBufferSwaps = acceleratedIterations;
        for (let iteration = 0; iteration < acceleratedIterations; iteration += 1) {
          pressure.setPipeline(this.iterateChebyshevPipeline);
          pressure.setBindGroup(0, iteration % 2 === 0 ? this.groups.ab : this.groups.ba);
          pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
        }
      } else {
        pressureBufferSwaps = this.iterations;
        for (let sweep = 0; sweep < this.iterations; sweep += 1) {
          const group = sweep % 2 === 0 ? this.groups.ab : this.groups.ba;
          gatherBodyCoupling(pressure, group);
          pressure.setPipeline(this.iteratePipeline); pressure.setBindGroup(0, group);
          pressure.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
        }
      }
      pressure.end();
    }
    // The megakernel folds its final iterate back into pressure A; the fixed-
    // count ladders land in A exactly when the sweep count is even.
    const finalInA = pressureBufferSwaps % 2 === 0;
    this.latestPressureInA = finalInA;
    const finalGroup = finalInA ? this.groups.ab : this.groups.ba;
    const project = encoder.beginComputePass({ label: "Octree finite-volume velocity projection", ...(detailedTimestampWrites?.projection ? { timestampWrites: detailedTimestampWrites.projection } : {}) });
    // The exact ladder refreshes M^-1 K^T p from the converged pressure so
    // projection sees the same-step solid response. The accelerated coupled
    // path intentionally skips that dependency: projection uses the rigid
    // velocity uploaded from the previous coupling batch, while the following
    // impulse pass publishes the current K^T p for the next batch.
    if (!useLaggedRigidCoupling) gatherBodyCoupling(project, finalGroup);
    if (this.couplingHasDynamicBodies && !useLaggedRigidCoupling) {
      project.setPipeline(this.applyBodyCouplingPipeline); project.setBindGroup(0, finalGroup); project.dispatchWorkgroups(1, 1, 1);
    }
    project.setPipeline(this.pressureImpulsePipeline);
    project.setBindGroup(0, finalInA ? this.couplingGroups.pressureA : this.couplingGroups.pressureB);
    project.dispatchWorkgroups(...this.workgroups);
    // A leaf-constant pressure basis constrains the integrated leaf flux but
    // has no gradient on the dense faces inside a coarse leaf. Reconstruct an
    // affine gradient into non-origin slots of the alternate pressure buffer;
    // project then applies it only to those internal faces, leaving every
    // solved coarse/fine boundary flux unchanged.
    project.setPipeline(this.reconstructSmallGradientsPipeline); project.setBindGroup(0, finalGroup);
    project.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
    project.setPipeline(this.reconstructGradientsPipeline); project.dispatchWorkgroupsIndirect(this.solveDispatch, 12);
    if (this.extrapolationSweeps > 0) {
      project.setPipeline(this.projectSmallLeavesPipeline); project.dispatchWorkgroupsIndirect(this.solveDispatch, 0);
      project.setPipeline(this.projectLeavesPipeline); project.dispatchWorkgroupsIndirect(this.solveDispatch, 12);
    } else {
      this.dispatch(project, this.projectPipeline, finalGroup);
    }
    project.end();
    for (let sweep = 0; sweep < this.extrapolationSweeps; sweep += 1) {
      const timing = detailedTimestampWrites?.extrapolation;
      const extrapolationTimestampWrites = timing && (sweep === 0 || sweep === this.extrapolationSweeps - 1) ? {
        querySet: timing.querySet,
        ...(sweep === 0 && timing.beginningOfPassWriteIndex !== undefined ? { beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex } : {}),
        ...(sweep === this.extrapolationSweeps - 1 && timing.endOfPassWriteIndex !== undefined ? { endOfPassWriteIndex: timing.endOfPassWriteIndex } : {})
      } : undefined;
      const extrapolate = encoder.beginComputePass({ label: "Octree narrow-band velocity extrapolation", ...(extrapolationTimestampWrites ? { timestampWrites: extrapolationTimestampWrites } : {}) });
      this.dispatch(extrapolate, sweep === 0 ? this.extrapolateSeedPipeline : this.extrapolatePipeline, sweep % 2 === 0 ? this.groups.extrapolateOut : this.groups.extrapolateScratch);
      extrapolate.end();
    }
    // An odd sweep ends in scratch; copy it back to the solver's authoritative velocity.
    if (this.extrapolationSweeps % 2 === 1) encoder.copyTextureToTexture({ texture: this.resources.velocityScratch }, { texture: this.resources.velocityOut }, [this.dims.nx, this.dims.ny, this.dims.nz]);
    this.encodeOverlayMaterialization(encoder, finalInA, detailedTimestampWrites?.materialization);
  }

  /** Publish lazily allocated diagnostic textures from the live owner map.
   * The first overlay request materializes immediately, so reset-time grid
   * inspection never decodes zero-initialized topology storage as finest 1^3. */
  encodeOverlayMaterialization(encoder: GPUCommandEncoder, pressureInA = this.latestPressureInA, timestampWrites?: GPUComputePassTimestampWrites) {
    if (!this.diagnosticGroups) return false;
    const materialize = encoder.beginComputePass({ label: "Materialize octree overlay fields", ...(timestampWrites ? { timestampWrites } : {}) });
    materialize.setPipeline(this.materializePipeline);
    materialize.setBindGroup(0, pressureInA ? this.diagnosticGroups.pressureA : this.diagnosticGroups.pressureB);
    materialize.dispatchWorkgroups(...this.workgroups);
    materialize.end();
    return true;
  }

  encodeSurface(encoder: GPUCommandEncoder, dt_s: number, inflow?: SurfaceInflowState, _maximumDt_s?: number, timestampWrites?: GPUComputePassTimestampWrites) {
    this.surfaceState.encode(encoder, dt_s, inflow, timestampWrites);
    this.encodeSurfaceBand(encoder, dt_s);
  }
  encodeSurfaceBand(encoder: GPUCommandEncoder, dt_s: number) {
    this.sparseSurfaceBand?.encode(encoder, dt_s);
  }
  addSurfaceReferenceVolumeCells(cells: number) { this.surfaceState.addReferenceVolumeCells(cells); }
  async readSolveDiagnostics() {
    // The compaction header is already the authoritative GPU result for this
    // solve. Sample its first two words only when the existing diagnostics
    // poll runs, so the simulation queue never waits on the CPU for topology.
    const readback = this.device.createBuffer({
      label: "Octree live pressure-row diagnostics",
      size: 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read octree pressure-row diagnostics" });
    encoder.copyBufferToBuffer(this.compaction, 0, readback, 0, 8);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange(0, 8));
      const liquidRows = words[0];
      this.info.pressureSampleCount = liquidRows;
      this.info.liquidDofCount = liquidRows;
      this.info.compressionRatio = liquidRows / Math.max(1, this.dims.nx * this.dims.ny * this.dims.nz);
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }
  get surfaceDiagnostics() { return this.surfaceState.volumeDiagnostics; }
  readSurfaceDiagnostics() { return this.surfaceState.readVolumeDiagnostics(); }
  encodeBodyImpulseReadback(_encoder: GPUCommandEncoder) { return undefined; }
  readBodyImpulseReadback(_buffer: GPUBuffer) { return Promise.resolve([]); }
  destroySharedSurface() { /* The octree owns its surface for its full lifetime. */ }
  get sparseVoxelRenderSource() { return this.sparseBrickWorld.renderSource; }
  get sparseSurfaceBandSource(): SparseSurfaceBandGPUSource | undefined { return this.sparseSurfaceBand?.source; }
  get sparseSurfaceRefinementFactor() { return this.sparseSurfaceBand?.plan.refinementFactor ?? 1; }
  get requiresFineSurfaceTimestep() { return this.sparseSurfaceBand?.requiresFineTimestep ?? false; }
  get fluidBrickCapacity() { return this.sparseBrickWorld.residency.capacity; }
  readFluidBrickResidencyStats() { return this.sparseBrickWorld.readResidencyStats(); }
  readFluidBrickAtlasStats() { return this.sparseBrickWorld.readAtlasStats(); }
  readSparseSurfaceBandStats() { return this.sparseSurfaceBand?.readStats(); }
  encodeSparseBrickWorld(encoder: GPUCommandEncoder, timings: {
    residency?: GPUComputePassTimestampWrites;
    publication?: GPUComputePassTimestampWrites;
  } = {}, dt_s = 0) {
    this.sparseBrickWorld.encode(encoder, {
      levelSet: this.levelSetTexture,
      velocity: this.resources.velocityOut,
      solidCells: this.solidCells
    }, timings, dt_s);
    this.topologyWorklistReady = true;
  }

  destroy() {
    this.topology.destroy(); this.pressureA.destroy(); this.pressureB.destroy(); this.params.destroy();
    this.topologyCandidateDispatch.destroy();
    this.compaction.destroy(); this.leafHeaders.destroy(); this.leafEntries.destroy(); this.leafFrontier.destroy(); this.solveDispatch.destroy(); this.solidCells.destroy();
    this.topologyDiagnosticTexture?.destroy(); this.pressureSamplesDiagnosticTexture?.destroy(); this.pressureDiagnosticTexture?.destroy(); this.divergenceDiagnosticTexture?.destroy();
    this.surfaceState.destroy();
    this.sparseSurfaceBand?.destroy();
    this.sparseBrickWorld.destroy();
  }
}

function initialOctreeLevelSet(
  scene: SceneDescription,
  dims: { nx: number; ny: number; nz: number },
  cell: { x: number; y: number; z: number }
) {
  const { nx, ny, nz } = dims;
  const alpha = new Float32Array(nx * ny * nz);
  const dam = damBreakFractions(scene.container.fillFraction);
  const terrainHeights = terrainColumnHeights(scene, nx, nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const aboveGround = (y + 0.5) * cell.y > terrainHeights[x + nx * z];
    const brickWet = initialFluidBrickContainsCell(scene, x, y, z, [nx, ny, nz]);
    const wet = aboveGround && combineInitialBrickWet(scene, brickWet, scene.fluid.initialCondition === "dam-break"
      ? (x + 0.5) / nx <= dam.width && (y + 0.5) / ny <= dam.height && (z + 0.5) / nz <= dam.depth
      : (y + 0.5) / ny <= scene.container.fillFraction);
    alpha[x + nx * (y + ny * z)] = wet ? 1 : 0;
  }
  return signedDistanceFromVolume(alpha, nx, ny, nz, cell);
}

export const octreeProjectionShader = /* wgsl */ `
override targetRefinementSize: u32 = 0u;

struct Owner { packedOrigin: u32, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f }
struct LeafHeader { cell: u32, entryStart: u32, entryCount: u32, size: u32, diagonal: f32, rhs: f32, pad0: u32, pad1: u32 }
struct LeafEntry { cell: u32, coefficient: f32 }
struct RigidBody { positionShape: vec4f, dimensions: vec4f, orientation: vec4f, linearVelocity: vec4f, angularVelocity: vec4f, inverseMassInertia: vec4f, angularMomentumRestitution: vec4f, material: vec4f }
struct SolidCell { fraction: f32, owner: i32 }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
// [0] = row count, [1] = entry count, [2..4] = row-parallel indirect args,
// [5..7] = cooperative coarse-assembly indirect args, [8] = coarse row count,
// [9..11] = one-workgroup-per-leaf args, [12..14] = frontier row-plan args;
// per-block (rows, entries, coarse rows) totals (later exclusive offsets) start
// at word 15. The dispatch words are copied out after their producing pass because one
// buffer cannot be writable storage and indirect in the same dispatch scope.
@group(0) @binding(2) var<storage, read_write> compaction: array<u32>;
@group(0) @binding(3) var<storage, read_write> owners: array<Owner>;
@group(0) @binding(4) var<storage, read_write> pressureIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var levelSetIn: texture_3d<f32>;
@group(0) @binding(8) var<storage, read_write> leafHeaders: array<LeafHeader>;
@group(0) @binding(9) var<storage, read_write> leafEntries: array<LeafEntry>;
@group(0) @binding(10) var<storage, read_write> rigidBodies: array<RigidBody, 12>;
@group(0) @binding(11) var<storage, read_write> solidCells: array<SolidCell>;
@group(0) @binding(12) var terrainIn: texture_2d<f32>;
// [0..1] ping-pong counts, [2] current-list selector, [3] reserved,
// followed by list A, list B, and one alive word per finest-cell origin.
@group(0) @binding(13) var<storage, read_write> frontier: array<atomic<u32>>;

fn dims() -> vec3u { return params.dimsMax.xyz; }
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn packOrigin(p: vec3u) -> u32 { return p.x | (p.y << 10u) | (p.z << 20u); }
fn unpackOrigin(word: u32) -> vec3u { return vec3u(word & 1023u, (word >> 10u) & 1023u, (word >> 20u) & 1023u); }
fn ownerAt(p: vec3i) -> Owner { return owners[index(vec3u(p))]; }
fn phi(p: vec3i) -> f32 { if (!valid(p)) { return 3.402823e38; } return textureLoad(levelSetIn, p, 0).x; }
fn liquidCell(p: vec3i) -> bool { return valid(p) && phi(p) < 0.0; }
fn ownerPhi(owner: Owner) -> f32 {
  // Pressure lives at the geometric leaf centre. Even-sized leaves therefore
  // sit between fine-cell samples; trilinear reconstruction avoids the
  // upper-corner classification bias of origin + size/2.
  let centre = vec3f(unpackOrigin(owner.packedOrigin)) + vec3f(0.5 * f32(owner.size - 1u));
  let a = vec3u(floor(centre)); let b = min(a + vec3u(1u), dims() - vec3u(1u)); let t = fract(centre);
  let p000 = phi(vec3i(a)); let p100 = phi(vec3i(vec3u(b.x,a.y,a.z)));
  let p010 = phi(vec3i(vec3u(a.x,b.y,a.z))); let p110 = phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001 = phi(vec3i(vec3u(a.x,a.y,b.z))); let p101 = phi(vec3i(vec3u(b.x,a.y,b.z)));
  let p011 = phi(vec3i(vec3u(a.x,b.y,b.z))); let p111 = phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn liquidOwner(owner: Owner) -> bool {
  return ownerPhi(owner) < 0.0;
}
fn isOrigin(id: vec3u, owner: Owner) -> bool { return all(id == unpackOrigin(owner.packedOrigin)); }
fn cellCount() -> u32 { return params.dimsMax.x * params.dimsMax.y * params.dimsMax.z; }
fn frontierBase(which: u32) -> u32 { return 4u + which * cellCount(); }
fn frontierAliveBase() -> u32 { return 4u + 2u * cellCount(); }
fn frontierCurrent() -> u32 { return atomicLoad(&frontier[2]); }
fn frontierGeneration() -> u32 { return atomicLoad(&frontier[3]); }
fn frontierCount(which: u32) -> u32 { return min(atomicLoad(&frontier[which]), cellCount()); }
fn frontierCell(which: u32, slot: u32) -> u32 { return atomicLoad(&frontier[frontierBase(which) + slot]); }
fn frontierAlive(cell: u32) -> bool { return atomicLoad(&frontier[frontierAliveBase() + cell]) != 0u; }
fn frontierInvalidate(cell: u32) { atomicStore(&frontier[frontierAliveBase() + cell], 0u); }
fn frontierAppend(which: u32, cell: u32) {
  if (atomicExchange(&frontier[frontierAliveBase() + cell], 1u) != 0u) { return; }
  let slot = atomicAdd(&frontier[which], 1u);
  if (slot < cellCount()) { atomicStore(&frontier[frontierBase(which) + slot], cell); }
}
fn component(v: vec3f, axis: u32) -> f32 { return select(select(v.z, v.y, axis == 1u), v.x, axis == 0u); }
fn axisVector(axis: u32) -> vec3i { return select(select(vec3i(0,0,1), vec3i(0,1,0), axis == 1u), vec3i(1,0,0), axis == 0u); }
fn faceArea(axis: u32) -> f32 {
  let h = params.cellRelax.xyz;
  return select(select(h.x * h.y, h.x * h.z, axis == 1u), h.y * h.z, axis == 0u);
}
fn cellWidth(axis: u32) -> f32 { return component(params.cellRelax.xyz, axis); }
fn pressureDistance(a: Owner, b: Owner, axis: u32) -> f32 {
  let full = 0.5 * f32(a.size + b.size) * cellWidth(axis);
  let phiA = ownerPhi(a); let phiB = ownerPhi(b);
  if ((phiA < 0.0) == (phiB < 0.0)) { return full; }
  // Ghost-fluid/Ando--Batty distance: p=0 lies at the zero crossing of the
  // resident level set, not at the neighbouring air leaf centre. The lower
  // bound is a geometric degeneracy guard equivalent to the quadtree's
  // bounded free-surface weight, not a pressure tuning coefficient.
  let liquidPhi = select(phiB, phiA, phiA < 0.0);
  let airPhi = select(phiA, phiB, phiA < 0.0);
  let theta = clamp(abs(liquidPhi) / max(abs(liquidPhi) + abs(airPhi), 1e-12), 0.01, 1.0);
  return theta * full;
}
fn worldCell(p: vec3i) -> vec3f {
  let h = params.cellRelax.xyz;
  return vec3f(-0.5 * params.container.x + (f32(p.x) + 0.5) * h.x, (f32(p.y) + 0.5) * h.y, -0.5 * params.container.z + (f32(p.z) + 0.5) * h.z);
}
fn quaternionRotate(q: vec4f, v: vec3f) -> vec3f { let uv = cross(q.yzw, v); let uuv = cross(q.yzw, uv); return v + 2.0 * (q.x * uv + uuv); }
fn quaternionInverseRotate(q: vec4f, v: vec3f) -> vec3f { return quaternionRotate(vec4f(q.x, -q.yzw), v); }
fn insideRigid(body: RigidBody, world: vec3f) -> bool {
  let p = quaternionInverseRotate(body.orientation, world - body.positionShape.xyz); let d = body.dimensions.xyz; let shape = i32(round(body.positionShape.w));
  if (shape == 0) { return length(p) <= d.x; }
  if (shape == 1) { return all(abs(p) <= 0.5 * d); }
  if (shape == 2) { let cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y); return length(vec3f(p.x, p.y - cy, p.z)) <= d.x; }
  return p.x * p.x + p.z * p.z <= d.x * d.x && abs(p.y) <= 0.5 * d.y;
}
fn insideInflowChannel(world: vec3f) -> bool {
  if (params.inflowPositionRadius.w <= 0.0 || params.inflowDirectionLength.w <= 0.0) { return false; }
  let delta = world - params.inflowPositionRadius.xyz;
  let along = dot(delta, params.inflowDirectionLength.xyz);
  let radial = delta - along * params.inflowDirectionLength.xyz;
  let margin = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  return abs(along) <= 0.5 * params.inflowDirectionLength.w + margin && length(radial) <= params.inflowPositionRadius.w + margin;
}
fn bodySolidFraction(body: RigidBody, p: vec3i) -> f32 {
  let center = worldCell(p); let h = params.cellRelax.xyz; var inside = 0.0;
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let offset = vec3f(select(-0.4, 0.4, (corner & 1u) != 0u), select(-0.4, 0.4, (corner & 2u) != 0u), select(-0.4, 0.4, (corner & 4u) != 0u));
    if (insideRigid(body, center + offset * h)) { inside += 1.0; }
  }
  return inside / 8.0;
}
fn solidAt(p: vec3i) -> SolidCell { if (!valid(p)) { return SolidCell(1.0, -1); } return solidCells[index(vec3u(p))]; }
fn faceSolid(a: vec3i, b: vec3i) -> SolidCell { let sa = solidAt(a); let sb = solidAt(b); if (sa.fraction >= sb.fraction) { return sa; } return sb; }
fn faceWorld(faceCell: vec3i, axis: u32) -> vec3f { var result = worldCell(faceCell); result[axis] += 0.5 * params.cellRelax[axis]; return result; }
fn solidVelocity(solid: SolidCell, world: vec3f) -> vec3f {
  if (solid.owner < 0) { return vec3f(0.0); }
  let body = rigidBodies[u32(solid.owner)];
  return body.linearVelocity.xyz + cross(body.angularVelocity.xyz, world - body.positionShape.xyz);
}
fn couplingBase() -> u32 { return 15u + 3u * params.control.z; }
fn coarseTaskListBase() -> u32 { return couplingBase() + 12u * 8u; }
fn coarseTaskCapacity() -> u32 { return (cellCount() + 511u) / 512u; }
fn coarseTaskRow(task: u32) -> u32 { return compaction[coarseTaskListBase() + 2u * task]; }
fn coarseTaskTile(task: u32) -> u32 { return compaction[coarseTaskListBase() + 2u * task + 1u]; }
fn couplingAcceleration(body: u32, component: u32) -> f32 { return bitcast<f32>(compaction[couplingBase() + body * 8u + component]); }
fn constrainedFaceVelocity(faceCell: vec3i, axis: u32, solid: SolidCell) -> f32 {
  let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
  return open * component(velocityAt(faceCell), axis) + solid.fraction * component(solidVelocity(solid, faceWorld(faceCell, axis)), axis);
}

// Topology-tile worklist header occupies words 0..15 of the copied buffer:
// word 0 the active tile count, word 1 the active dispatch x width, word 4
// and word 5 the retired equivalents. A tile spans max(8, maximumLeaf) cells
// per axis so every dyadic pressure leaf lies inside exactly one tile; each
// tile decomposes into (tileSize/4)^3 of the existing 4^3 cell workgroups.
fn topologyTileSize() -> u32 { return max(8u, params.dimsMax.w); }
fn topologyTileCell(workgroup: vec3u, local: vec3u, countWord: u32, widthWord: u32, indexBase: u32) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = tileSize / 4u;
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[widthWord];
  let streamIndex = linearWorkgroup / groupsPerTile;
  // The 2D-tiled indirect dispatch may round up; out-of-list workgroups map
  // to an out-of-domain cell that every kernel's bounds guard rejects.
  if (streamIndex >= compaction[countWord]) { return vec3u(0xffffffffu); }
  let sub = linearWorkgroup % groupsPerTile;
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tileIndex = compaction[indexBase + streamIndex];
  let tile = vec3u(tileIndex % tx, (tileIndex / tx) % ty, tileIndex / (tx * ty));
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 4u + local;
}

fn residentTopologyCell(workgroup: vec3u, local: vec3u) -> vec3u {
  return topologyTileCell(workgroup, local, 0u, 1u, 16u);
}

// Retired tiles hold just-retired bricks with no resident sibling. Resetting
// and rebuilding them prevents old interface leaves from fossilizing outside
// residency; their indices follow the active capacity in the copied list.
fn retiredTopologyCell(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return topologyTileCell(workgroup, local, 4u, 5u, 16u + tx * ty * tz);
}

// Refinement and balancing can only act on leaves of size >= 2. Their origins
// are even-aligned, so candidate passes cover an 8^3 cell region with each
// 4^3 workgroup instead of launching one invocation for every finest cell.
fn topologyCandidateCell(workgroup: vec3u, local: vec3u, countWord: u32, widthWord: u32, indexBase: u32) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = max(1u, tileSize / 8u);
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[widthWord];
  let streamIndex = linearWorkgroup / groupsPerTile;
  if (streamIndex >= compaction[countWord]) { return vec3u(0xffffffffu); }
  let sub = linearWorkgroup % groupsPerTile;
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tileIndex = compaction[indexBase + streamIndex];
  let tile = vec3u(tileIndex % tx, (tileIndex / tx) % ty, tileIndex / (tx * ty));
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 8u + local * 2u;
}

fn residentTopologyCandidate(workgroup: vec3u, local: vec3u) -> vec3u {
  return topologyCandidateCell(workgroup, local, 0u, 9u, 16u);
}

fn retiredTopologyCandidate(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return topologyCandidateCell(workgroup, local, 4u, 13u, 16u + tx * ty * tz);
}

fn rasterizeSolidsAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let p = vec3i(gid); var fraction = 0.0; var owner = -1;
  if (params.container.w > 0.5) { fraction = clamp(textureLoad(terrainIn, vec2i(p.x, p.z), 0).x - f32(p.y), 0.0, 1.0); }
  if (!insideInflowChannel(worldCell(p))) {
    for (var bodyIndex = 0u; bodyIndex < 12u; bodyIndex += 1u) {
      if (bodyIndex >= params.control.w) { break; }
      let candidate = bodySolidFraction(rigidBodies[bodyIndex], p);
      if (candidate > fraction) { fraction = candidate; owner = i32(bodyIndex); }
    }
  }
  solidCells[index(gid)] = SolidCell(fraction, owner);
}

@compute @workgroup_size(4,4,4)
fn rasterizeSolids(@builtin(global_invocation_id) gid: vec3u) { rasterizeSolidsAt(gid); }

@compute @workgroup_size(4,4,4)
fn rasterizeSolidsActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  rasterizeSolidsAt(residentTopologyCell(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn rasterizeSolidsRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  rasterizeSolidsAt(retiredTopologyCell(wid, lid));
}

fn resetTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  var size = params.dimsMax.w;
  var origin = (gid / vec3u(size)) * vec3u(size);
  loop {
    if (all(origin + vec3u(size) <= dims()) || size == 1u) { break; }
    size = size / 2u; origin = (gid / vec3u(size)) * vec3u(size);
  }
  owners[index(gid)] = Owner(packOrigin(origin), size);
}

fn invalidateFrontierAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let old = owners[index(gid)];
  if (old.size > 0u && isOrigin(gid, old)) { frontierInvalidate(index(gid)); }
}

@compute @workgroup_size(4,4,4)
fn resetTopology(@builtin(global_invocation_id) gid: vec3u) { resetTopologyAt(gid); }

@compute @workgroup_size(4,4,4)
fn resetTopologyActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let gid = residentTopologyCell(wid, lid); invalidateFrontierAt(gid); resetTopologyAt(gid);
}

@compute @workgroup_size(4,4,4)
fn resetTopologyRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let gid = retiredTopologyCell(wid, lid); invalidateFrontierAt(gid); resetTopologyAt(gid);
}

fn leafNeedsRefinement(origin: vec3u, size: u32) -> bool {
  var closestSurface = 3.402823e38; var minimumPhi = 3.402823e38; var maximumPhi = -3.402823e38; var minimumSolid = 1.0; var maximumSolid = 0.0;
  var minimumSurfaceVelocity = vec3f(0.0); var maximumSurfaceVelocity = vec3f(0.0); var maximumCurvatureProxy = 0.0; var hasSurfaceSample = false;
  let finestWidth = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  let baseBand = max(0.0, params.solve.w);
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q = origin + vec3u(x,y,z); let solid = solidCells[index(q)].fraction;
    let samplePhi = phi(vec3i(q));
    closestSurface = min(closestSurface, abs(samplePhi)); minimumPhi = min(minimumPhi, samplePhi); maximumPhi = max(maximumPhi, samplePhi); minimumSolid = min(minimumSolid, solid); maximumSolid = max(maximumSolid, solid);
    if (params.physical.w > 0.0 && abs(samplePhi) < (baseBand + 2.0) * finestWidth) {
      let velocity = textureLoad(velocityIn, vec3i(q), 0).xyz;
      if (!hasSurfaceSample) { minimumSurfaceVelocity = velocity; maximumSurfaceVelocity = velocity; }
      else { minimumSurfaceVelocity = min(minimumSurfaceVelocity, velocity); maximumSurfaceVelocity = max(maximumSurfaceVelocity, velocity); }
      hasSurfaceSample = true;
      let p = vec3i(q);
      let laplacian = phi(p + vec3i(1,0,0)) + phi(p - vec3i(1,0,0))
        + phi(p + vec3i(0,1,0)) + phi(p - vec3i(0,1,0))
        + phi(p + vec3i(0,0,1)) + phi(p - vec3i(0,0,1)) - 6.0 * samplePhi;
      maximumCurvatureProxy = max(maximumCurvatureProxy, abs(laplacian) / max(finestWidth, 1e-6));
    }
  } } }
  // Interface-graded sizing: leaves that cross the liquid surface or a solid
  // boundary split, while uniform air, liquid, and solid bulk may stay coarse
  // once they are outside the explicit finest-cell interface band.
  let adaptivity = f32(params.control.x) / 1000.0;
  if (adaptivity <= 0.0) { return true; }
  let crossesSurface = minimumPhi < 0.0 && maximumPhi >= 0.0;
  let crossesSolidBoundary = maximumSolid - minimumSolid > 1e-5 || (maximumSolid > 1e-5 && maximumSolid < 1.0 - 1e-5);
  if (crossesSurface || crossesSolidBoundary) { return true; }
  if (minimumSolid >= 1.0 - 1e-5) { return false; }
  // This only widens the fine support band; it cannot coarsen a leaf selected
  // by the baseline signed-distance rule. Velocity span estimates deformation
  // over one configured maximum step, while the phi Laplacian detects curved
  // features. Keeping it inside the full-domain rebuild avoids stale brick
  // lists and preserves the dense transport/pressure authority.
  let strainActivity = select(0.0, length(maximumSurfaceVelocity - minimumSurfaceVelocity) * params.physical.z / max(finestWidth, 1e-6), hasSurfaceSample);
  let detailActivity = params.physical.w * clamp(max(strainActivity, 2.0 * maximumCurvatureProxy), 0.0, 1.0);
  let effectiveBand = baseBand + 8.0 * detailActivity;
  return closestSurface < effectiveBand * finestWidth;
}

fn splitLeaf(origin: vec3u, size: u32) {
  let child = size / 2u;
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let local = vec3u(x,y,z); let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    owners[index(origin + local)] = Owner(packOrigin(childOrigin), child);
  } } }
}

fn refineTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = owners[index(gid)];
  if (owner.size > 1u && (targetRefinementSize == 0u || owner.size == targetRefinementSize) && isOrigin(gid, owner) && leafNeedsRefinement(gid, owner.size)) { splitLeaf(gid, owner.size); }
}

@compute @workgroup_size(4,4,4)
fn refineTopology(@builtin(global_invocation_id) gid: vec3u) { refineTopologyAt(gid * 2u); }

@compute @workgroup_size(4,4,4)
fn refineTopologyActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  refineTopologyAt(residentTopologyCandidate(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn refineTopologyRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  refineTopologyAt(retiredTopologyCandidate(wid, lid));
}

// Large leaves are deliberately rare, so assigning their size^3 sizing scan
// and split to a single invocation starves the GPU. One 256-lane workgroup
// performs the exact same reduction and publishes child owners cooperatively.
var<workgroup> refineEligible: atomic<u32>;
var<workgroup> refineDecision: atomic<u32>;
var<workgroup> refineRange: array<vec4f, 256>;
var<workgroup> refineActivity: array<vec4f, 256>;
var<workgroup> refineMinVelocity: array<f32, 768>;
var<workgroup> refineMaxVelocity: array<f32, 768>;

@compute @workgroup_size(256)
fn refineTopologyCoarse(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let size = targetRefinementSize;
  let origin = wid * vec3u(size);
  if (lid == 0u) {
    let owner = owners[index(origin)];
    atomicStore(&refineEligible, select(0u, 1u, owner.size == size && isOrigin(origin, owner)));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&refineEligible) == 0u) { return; }

  var closestSurface = 3.402823e38;
  var minimumPhi = 3.402823e38;
  var maximumPhi = -3.402823e38;
  var minimumSolid = 1.0;
  var maximumSolid = 0.0;
  var minimumSurfaceVelocity = vec3f(3.402823e38);
  var maximumSurfaceVelocity = vec3f(-3.402823e38);
  var maximumCurvatureProxy = 0.0;
  var surfaceSamples = 0.0;
  let finestWidth = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  let baseBand = max(0.0, params.solve.w);
  let cells = size * size * size;
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let q = origin + local;
    let solid = solidCells[index(q)].fraction;
    let samplePhi = phi(vec3i(q));
    closestSurface = min(closestSurface, abs(samplePhi));
    minimumPhi = min(minimumPhi, samplePhi);
    maximumPhi = max(maximumPhi, samplePhi);
    minimumSolid = min(minimumSolid, solid);
    maximumSolid = max(maximumSolid, solid);
    if (params.physical.w > 0.0 && abs(samplePhi) < (baseBand + 2.0) * finestWidth) {
      let velocity = textureLoad(velocityIn, vec3i(q), 0).xyz;
      minimumSurfaceVelocity = min(minimumSurfaceVelocity, velocity);
      maximumSurfaceVelocity = max(maximumSurfaceVelocity, velocity);
      surfaceSamples += 1.0;
      let p = vec3i(q);
      let laplacian = phi(p + vec3i(1,0,0)) + phi(p - vec3i(1,0,0))
        + phi(p + vec3i(0,1,0)) + phi(p - vec3i(0,1,0))
        + phi(p + vec3i(0,0,1)) + phi(p - vec3i(0,0,1)) - 6.0 * samplePhi;
      maximumCurvatureProxy = max(maximumCurvatureProxy, abs(laplacian) / max(finestWidth, 1e-6));
    }
  }
  refineRange[lid] = vec4f(closestSurface, minimumPhi, maximumPhi, minimumSolid);
  refineActivity[lid] = vec4f(maximumSolid, maximumCurvatureProxy, surfaceSamples, 0.0);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    refineMinVelocity[3u * lid + axis] = minimumSurfaceVelocity[axis];
    refineMaxVelocity[3u * lid + axis] = maximumSurfaceVelocity[axis];
  }
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) {
      refineRange[lid].x = min(refineRange[lid].x, refineRange[lid + stride].x);
      refineRange[lid].y = min(refineRange[lid].y, refineRange[lid + stride].y);
      refineRange[lid].z = max(refineRange[lid].z, refineRange[lid + stride].z);
      refineRange[lid].w = min(refineRange[lid].w, refineRange[lid + stride].w);
      refineActivity[lid] = max(refineActivity[lid], refineActivity[lid + stride]);
      refineActivity[lid].z += refineActivity[lid + stride].z;
      for (var axis = 0u; axis < 3u; axis += 1u) {
        refineMinVelocity[3u * lid + axis] = min(refineMinVelocity[3u * lid + axis], refineMinVelocity[3u * (lid + stride) + axis]);
        refineMaxVelocity[3u * lid + axis] = max(refineMaxVelocity[3u * lid + axis], refineMaxVelocity[3u * (lid + stride) + axis]);
      }
    }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let range = refineRange[0];
    let activity = refineActivity[0];
    let adaptivity = f32(params.control.x) / 1000.0;
    let crossesSurface = range.y < 0.0 && range.z >= 0.0;
    let crossesSolidBoundary = activity.x - range.w > 1e-5 || (activity.x > 1e-5 && activity.x < 1.0 - 1e-5);
    var needsRefinement = adaptivity <= 0.0 || crossesSurface || crossesSolidBoundary;
    if (!needsRefinement && range.w < 1.0 - 1e-5) {
      let hasSurfaceSample = activity.z > 0.0;
      let velocitySpan = vec3f(
        refineMaxVelocity[0] - refineMinVelocity[0],
        refineMaxVelocity[1] - refineMinVelocity[1],
        refineMaxVelocity[2] - refineMinVelocity[2]
      );
      let strainActivity = select(0.0, length(velocitySpan) * params.physical.z / max(finestWidth, 1e-6), hasSurfaceSample);
      let detailActivity = params.physical.w * clamp(max(strainActivity, 2.0 * activity.y), 0.0, 1.0);
      needsRefinement = range.x < (baseBand + 8.0 * detailActivity) * finestWidth;
    }
    atomicStore(&refineDecision, select(0u, 1u, needsRefinement));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&refineDecision) == 0u) { return; }
  let child = size / 2u;
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    owners[index(origin + local)] = Owner(packOrigin(childOrigin), child);
  }
}

fn neighborTooFine(origin: vec3u, size: u32) -> bool {
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) {
    let q0 = vec3i(origin + vec3u(0u,y,z)); let q1 = vec3i(origin + vec3u(size-1u,y,z));
    if ((valid(q0-vec3i(1,0,0)) && ownerAt(q0-vec3i(1,0,0)).size * 2u < size) || (valid(q1+vec3i(1,0,0)) && ownerAt(q1+vec3i(1,0,0)).size * 2u < size)) { return true; }
  } }
  for (var z = 0u; z < size; z += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,0u,z)); let q1 = vec3i(origin + vec3u(x,size-1u,z));
    if ((valid(q0-vec3i(0,1,0)) && ownerAt(q0-vec3i(0,1,0)).size * 2u < size) || (valid(q1+vec3i(0,1,0)) && ownerAt(q1+vec3i(0,1,0)).size * 2u < size)) { return true; }
  } }
  for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,y,0u)); let q1 = vec3i(origin + vec3u(x,y,size-1u));
    if ((valid(q0-vec3i(0,0,1)) && ownerAt(q0-vec3i(0,0,1)).size * 2u < size) || (valid(q1+vec3i(0,0,1)) && ownerAt(q1+vec3i(0,0,1)).size * 2u < size)) { return true; }
  } }
  return false;
}

fn balanceTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = owners[index(gid)];
  // Size-16+ leaves use the cooperative entry point below.
  if (owner.size > 2u && owner.size < 16u && isOrigin(gid, owner) && neighborTooFine(gid, owner.size)) { splitLeaf(gid, owner.size); }
}


@compute @workgroup_size(4,4,4)
fn balanceTopology(@builtin(global_invocation_id) gid: vec3u) { balanceTopologyAt(gid * 2u); }

@compute @workgroup_size(4,4,4)
fn balanceTopologyActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  balanceTopologyAt(residentTopologyCandidate(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn balanceTopologyRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  balanceTopologyAt(retiredTopologyCandidate(wid, lid));
}

var<workgroup> balanceEligible: atomic<u32>;
var<workgroup> balanceFlags: array<u32, 256>;

@compute @workgroup_size(256)
fn balanceTopologyCoarse(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let size = targetRefinementSize;
  let origin = wid * vec3u(size);
  if (lid == 0u) {
    let owner = owners[index(origin)];
    atomicStore(&balanceEligible, select(0u, 1u, owner.size == size && isOrigin(origin, owner)));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&balanceEligible) == 0u) { return; }
  var needsSplit = 0u;
  let faceSamples = size * size;
  for (var sample = lid; sample < 6u * faceSamples; sample += 256u) {
    let face = sample / faceSamples;
    let axis = face / 2u;
    let positive = (face & 1u) == 1u;
    let within = sample % faceSamples;
    let a = within % size;
    let b = within / size;
    var local = vec3u(0u);
    local[axis] = select(0u, size - 1u, positive);
    local[(axis + 1u) % 3u] = a;
    local[(axis + 2u) % 3u] = b;
    let outside = vec3i(origin + local) + select(-1, 1, positive) * axisVector(axis);
    if (valid(outside) && ownerAt(outside).size * 2u < size) { needsSplit = 1u; }
  }
  balanceFlags[lid] = needsSplit;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { balanceFlags[lid] = max(balanceFlags[lid], balanceFlags[lid + stride]); }
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&balanceFlags[0]) == 0u) { return; }
  let cells = size * size * size;
  let child = size / 2u;
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    owners[index(origin + local)] = Owner(packOrigin(childOrigin), child);
  }
}

fn pressureOf(owner: Owner) -> f32 { return pressureIn[index(unpackOrigin(owner.packedOrigin))]; }
fn velocityAt(p: vec3i) -> vec3f { if (!valid(p)) { return vec3f(0.0); } return textureLoad(velocityIn, p, 0).xyz; }

fn inverseInertiaResponse(body: RigidBody, value: vec3f) -> vec3f {
  let local = quaternionInverseRotate(body.orientation, value);
  return quaternionRotate(body.orientation, local * body.inverseMassInertia.yzw);
}

// One row of K M^-1 K^T, where
// K = grad^T V (1-A) L. The first component is the current matrix action
// against the globally gathered M^-1 K^T p vector; the second is this row's
// exact rank-six diagonal, used by weighted Jacobi.
fn leafBodyCoupling(origin: vec3u, size: u32) -> vec2f {
  if (params.physical.y < 0.5) { return vec2f(0.0); }
  var linear: array<vec3f, 12>;
  var angular: array<vec3f, 12>;
  for (var face = 0u; face < 6u; face += 1u) {
    let axis = face / 2u; let side = select(-1, 1, (face & 1u) == 1u); let e = axisVector(axis); let area = faceArea(axis);
    for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
      var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local); let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let solid = faceSolid(inside, outside);
      if (solid.owner < 0 || u32(solid.owner) >= params.control.w || solid.fraction <= 0.0) { continue; }
      let bodyIndex = u32(solid.owner); let factor = -f32(side) * area * solid.fraction;
      var generator = vec3f(0.0); generator[axis] = factor;
      let faceCell = select(outside, inside, side > 0); let arm = faceWorld(faceCell, axis) - rigidBodies[bodyIndex].positionShape.xyz;
      linear[bodyIndex] += generator; angular[bodyIndex] += cross(arm, generator);
    } }
  }
  var action = 0.0; var diagonal = 0.0;
  for (var bodyIndex = 0u; bodyIndex < 12u; bodyIndex += 1u) {
    if (bodyIndex >= params.control.w) { break; }
    let body = rigidBodies[bodyIndex]; let linearGenerator = linear[bodyIndex]; let angularGenerator = angular[bodyIndex];
    let linearResponse = body.inverseMassInertia.x * linearGenerator;
    let angularResponse = inverseInertiaResponse(body, angularGenerator);
    diagonal += dot(linearGenerator, linearResponse) + dot(angularGenerator, angularResponse);
    let accelerated = vec3f(couplingAcceleration(bodyIndex, 0u), couplingAcceleration(bodyIndex, 1u), couplingAcceleration(bodyIndex, 2u));
    let angularAccelerated = vec3f(couplingAcceleration(bodyIndex, 3u), couplingAcceleration(bodyIndex, 4u), couplingAcceleration(bodyIndex, 5u));
    action += dot(linearGenerator, accelerated) + dot(angularGenerator, angularAccelerated);
  }
  return vec2f(action, diagonal);
}

fn accumulateFace(origin: vec3u, size: u32, axis: u32, side: i32, diagonal: ptr<function, f32>, sum: ptr<function, f32>, flux: ptr<function, f32>) {
  let e = axisVector(axis); let area = faceArea(axis);
  for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
    var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
    local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
    let inside = vec3i(origin + local); let outside = inside + side * e;
    if (!valid(outside)) { continue; }
    let neighbor = ownerAt(outside); let distance = pressureDistance(ownerAt(vec3i(origin)), neighbor, axis);
    let solid = faceSolid(inside, outside); let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
    let coefficient = open * area / max(distance, 1e-7);
    (*diagonal) += coefficient;
    if (liquidOwner(neighbor)) { (*sum) += coefficient * pressureOf(neighbor); }
    let faceCell = select(outside, inside, side > 0);
    (*flux) += f32(side) * area * constrainedFaceVelocity(faceCell, axis, solid);
  } }
}

@compute @workgroup_size(4,4,4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = owners[index(gid)]; let idx = index(gid);
  if (!isOrigin(gid, owner) || !liquidOwner(owner)) { pressureOut[idx] = 0.0; return; }
  var diagonal = 0.0; var sum = 0.0; var flux = 0.0;
  accumulateFace(gid, owner.size, 0u, -1, &diagonal, &sum, &flux); accumulateFace(gid, owner.size, 0u, 1, &diagonal, &sum, &flux);
  accumulateFace(gid, owner.size, 1u, -1, &diagonal, &sum, &flux); accumulateFace(gid, owner.size, 1u, 1, &diagonal, &sum, &flux);
  accumulateFace(gid, owner.size, 2u, -1, &diagonal, &sum, &flux); accumulateFace(gid, owner.size, 2u, 1, &diagonal, &sum, &flux);
  let coupling = leafBodyCoupling(gid, owner.size); let old = pressureIn[idx]; let effectiveDiagonal = diagonal + coupling.y;
  let next = select(0.0, old + (sum - flux - diagonal * old - coupling.x) / effectiveDiagonal, effectiveDiagonal > 1e-8);
  pressureOut[idx] = mix(pressureIn[idx], next, params.cellRelax.w);
}

// --- Compacted leaf solve -------------------------------------------------
// The dense jacobi above launches one thread per finest cell and rebuilds the
// entire row (coefficients, neighbor lookups, velocity flux) every sweep. The
// kernels below run the classic GPU-octree pipeline instead: a prefix-sum
// stream compaction of liquid leaf origins, a one-time assembly of each row's
// diagonal / RHS flux / merged neighbor table, then iteration kernels that
// only gather cached entries. 2:1 balance bounds a leaf's distinct neighbors
// at 4 per face (24 total; 6 for finest leaves), which also bounds the entry
// pool at 6 entries per finest cell.

// The leaf frontier survives topology rebuilds. Partial rebuilds invalidate
// only origins inside active/retired topology tiles; filtering preserves every
// untouched liquid leaf and tile-local append publishes the replacement
// origins. The dense finest lattice is scanned only for cold initialization.
@compute @workgroup_size(1)
fn beginFrontier() {
  let current = frontierCurrent();
  let next = 1u - current;
  atomicStore(&frontier[next], 0u);
  let blocks = (frontierCount(current) + 255u) / 256u;
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  compaction[12] = x; compaction[13] = y; compaction[14] = 1u;
}

@compute @workgroup_size(256)
fn filterFrontier(@builtin(global_invocation_id) gid: vec3u) {
  let current = frontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  if (slot >= frontierCount(current)) { return; }
  let cell = frontierCell(current, slot);
  if (frontierAlive(cell)) {
    let next = 1u - current;
    let output = atomicAdd(&frontier[next], 1u);
    if (output < cellCount()) { atomicStore(&frontier[frontierBase(next) + output], cell); }
  }
}

fn appendFrontierAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let cell = index(gid);
  let owner = owners[cell];
  if (isOrigin(gid, owner) && liquidOwner(owner)) { frontierAppend(1u - frontierCurrent(), cell); }
}

@compute @workgroup_size(4,4,4)
fn appendFrontier(@builtin(global_invocation_id) gid: vec3u) { appendFrontierAt(gid); }

@compute @workgroup_size(4,4,4)
fn appendFrontierActive(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  appendFrontierAt(residentTopologyCell(wid, lid));
}

@compute @workgroup_size(4,4,4)
fn appendFrontierRetired(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  appendFrontierAt(retiredTopologyCell(wid, lid));
}

@compute @workgroup_size(1)
fn finalizeFrontier() {
  let next = 1u - frontierCurrent();
  let count = frontierCount(next);
  atomicStore(&frontier[2], next);
  atomicAdd(&frontier[3], 1u);
  let blocks = (count + 255u) / 256u;
  compaction[8] = blocks;
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  compaction[12] = x; compaction[13] = y; compaction[14] = 1u;
}

fn cellCoord(c: u32) -> vec3u {
  let nx = params.dimsMax.x; let ny = params.dimsMax.y;
  return vec3u(c % nx, (c / nx) % ny, c / (nx * ny));
}

var<workgroup> bodyCouplingScratch: array<f32, 256>;
var<workgroup> bodyGeneralized: array<f32, 6>;

// One workgroup per body evaluates K^T p over every finest positive face,
// then stores rho M^-1 K^T p in the dead tail of the compaction buffer. A
// following dispatch applies the rank-six response without atomics or a host
// synchronization point.
@compute @workgroup_size(256)
fn gatherBodyCoupling(@builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x; let bodyIndex = wid.x;
  if (bodyIndex >= params.control.w) { return; }
  let body = rigidBodies[bodyIndex];
  var force = vec3f(0.0); var torque = vec3f(0.0);
  for (var flat = lid; flat < cellCount() * 3u; flat += 256u) {
    let axis = flat % 3u; let p = vec3i(cellCoord(flat / 3u)); let q = p + axisVector(axis);
    if (!valid(q)) { continue; }
    let solid = faceSolid(p, q);
    if (solid.owner < 0 || u32(solid.owner) != bodyIndex || solid.fraction <= 0.0) { continue; }
    let left = ownerAt(p); let right = ownerAt(q); let leftWet = liquidOwner(left); let rightWet = liquidOwner(right);
    if (!leftWet && !rightWet) { continue; }
    let p0 = select(0.0, pressureOf(left), leftWet); let p1 = select(0.0, pressureOf(right), rightWet);
    let magnitude = faceArea(axis) * solid.fraction * (p1 - p0);
    var generator = vec3f(0.0); generator[axis] = magnitude;
    force += generator; torque += cross(faceWorld(p, axis) - body.positionShape.xyz, generator);
  }
  let values = array<f32, 6>(force.x, force.y, force.z, torque.x, torque.y, torque.z);
  for (var component = 0u; component < 6u; component += 1u) {
    bodyCouplingScratch[lid] = values[component];
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) { bodyCouplingScratch[lid] += bodyCouplingScratch[lid + stride]; }
    }
    workgroupBarrier();
    if (lid == 0u) { bodyGeneralized[component] = bodyCouplingScratch[0]; }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let linear = body.inverseMassInertia.x * vec3f(bodyGeneralized[0], bodyGeneralized[1], bodyGeneralized[2]);
    let angular = inverseInertiaResponse(body, vec3f(bodyGeneralized[3], bodyGeneralized[4], bodyGeneralized[5]));
    let response = array<f32, 6>(linear.x, linear.y, linear.z, angular.x, angular.y, angular.z);
    for (var component = 0u; component < 6u; component += 1u) { compaction[couplingBase() + bodyIndex * 8u + component] = bitcast<u32>(response[component]); }
  }
}

// Commit the same-step pressure response to the GPU-resident body velocity.
// Subsequent substeps and the shared tangential Brinkman pass therefore see
// the same state whose equal-and-opposite impulse is returned to the CPU.
@compute @workgroup_size(64)
fn applyBodyCoupling(@builtin(global_invocation_id) gid: vec3u) {
  let bodyIndex = gid.x;
  if (params.physical.y < 0.5 || bodyIndex >= params.control.w) { return; }
  var body = rigidBodies[bodyIndex];
  let linear = vec3f(couplingAcceleration(bodyIndex, 0u), couplingAcceleration(bodyIndex, 1u), couplingAcceleration(bodyIndex, 2u));
  let angular = vec3f(couplingAcceleration(bodyIndex, 3u), couplingAcceleration(bodyIndex, 4u), couplingAcceleration(bodyIndex, 5u));
  body.linearVelocity = vec4f(body.linearVelocity.xyz - linear, body.linearVelocity.w);
  body.angularVelocity = vec4f(body.angularVelocity.xyz - angular, body.angularVelocity.w);
  rigidBodies[bodyIndex] = body;
}

fn leafInfo(c: u32) -> vec3u {
  let owner = owners[c];
  if (!frontierAlive(c) || !isOrigin(cellCoord(c), owner)) { return vec3u(0u); }
  var coarseTasks = 0u;
  if (owner.size >= 8u) { let tiles = owner.size / 8u; coarseTasks = tiles * tiles * tiles; }
  return vec3u(1u, select(24u, 6u, owner.size == 1u), coarseTasks);
}

var<workgroup> scanPairs: array<vec3u, 256>;

@compute @workgroup_size(256)
fn planLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  var value = vec3u(0u);
  let current = frontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  if (slot < frontierCount(current)) { value = leafInfo(frontierCell(current, slot)); }
  scanPairs[lid] = value;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { scanPairs[lid] += scanPairs[lid + stride]; }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let total = scanPairs[0];
    let block = wid.x + wid.y * compaction[12];
    compaction[15u + 3u * block] = total.x;
    compaction[16u + 3u * block] = total.y;
    compaction[17u + 3u * block] = total.z;
  }
}

@compute @workgroup_size(256)
fn scanLeafBlocks(@builtin(local_invocation_id) lid3: vec3u) {
  let lid = lid3.x;
  let blocks = compaction[8];
  let chunk = (blocks + 255u) / 256u;
  let base = lid * chunk;
  var sum = vec3u(0u);
  for (var i = 0u; i < chunk; i += 1u) {
    let b = base + i;
    if (b < blocks) { sum += vec3u(compaction[15u + 3u * b], compaction[16u + 3u * b], compaction[17u + 3u * b]); }
  }
  scanPairs[lid] = sum;
  for (var stride = 1u; stride < 256u; stride <<= 1u) {
    workgroupBarrier();
    var add = vec3u(0u);
    if (lid >= stride) { add = scanPairs[lid - stride]; }
    workgroupBarrier();
    scanPairs[lid] += add;
  }
  workgroupBarrier();
  var running = scanPairs[lid] - sum;
  for (var i = 0u; i < chunk; i += 1u) {
    let b = base + i;
    if (b < blocks) {
      let pair = vec3u(compaction[15u + 3u * b], compaction[16u + 3u * b], compaction[17u + 3u * b]);
      compaction[15u + 3u * b] = running.x;
      compaction[16u + 3u * b] = running.y;
      compaction[17u + 3u * b] = running.z;
      running += pair;
    }
  }
  if (lid == 255u) {
    let total = scanPairs[255];
    compaction[0] = total.x; compaction[1] = total.y;
    let blocks = (total.x + 255u) / 256u;
    let x = min(blocks, 65535u);
    var y = 1u;
    if (x > 0u) { y = (blocks + x - 1u) / x; }
    compaction[2] = x; compaction[3] = y; compaction[4] = 1u;
    // Coarse velocity work is tiled into 8^3 chunks. This both bounds the work
    // per workgroup and restores occupancy for a handful of very large leaves.
    let cooperativeTasks = min(total.z, 65535u);
    compaction[5] = cooperativeTasks; compaction[6] = 1u; compaction[7] = 1u; compaction[8] = cooperativeTasks;
    let leafX = min(total.x, 65535u);
    var leafY = 1u;
    if (leafX > 0u) { leafY = (total.x + leafX - 1u) / leafX; }
    compaction[9] = leafX; compaction[10] = leafY; compaction[11] = 1u;
  }
}

@compute @workgroup_size(256)
fn emitLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  var value = vec3u(0u);
  let current = frontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  var cell = 0u;
  if (slot < frontierCount(current)) { cell = frontierCell(current, slot); value = leafInfo(cell); }
  scanPairs[lid] = value;
  for (var stride = 1u; stride < 256u; stride <<= 1u) {
    workgroupBarrier();
    var add = vec3u(0u);
    if (lid >= stride) { add = scanPairs[lid - stride]; }
    workgroupBarrier();
    scanPairs[lid] += add;
  }
  workgroupBarrier();
  if (value.x == 1u) {
    let exclusive = scanPairs[lid] - value;
    let block = wid.x + wid.y * compaction[12];
    let row = compaction[15u + 3u * block] + exclusive.x;
    let start = compaction[16u + 3u * block] + exclusive.y;
    let taskStart = compaction[17u + 3u * block] + exclusive.z;
    let cooperative = select(0u, 1u, value.z > 0u && taskStart + value.z <= coarseTaskCapacity());
    leafHeaders[row] = LeafHeader(cell, start, 0u, owners[cell].size, 0.0, 0.0, cooperative, 0u);
    if (cooperative == 1u) {
      for (var tile = 0u; tile < value.z; tile += 1u) {
        let task = taskStart + tile;
        compaction[coarseTaskListBase() + 2u * task] = row;
        compaction[coarseTaskListBase() + 2u * task + 1u] = tile;
      }
    }
  }
}

fn compactRowIndex(gid: vec3u) -> u32 { return gid.x + gid.y * compaction[2] * 256u; }

@compute @workgroup_size(256)
fn assembleSystem(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  var header = leafHeaders[row];
  // Size>=8 rows are emitted to a separate deterministic stream and assembled
  // by one cooperative workgroup each. Tiny rows retain this occupancy-friendly
  // one-invocation path instead of paying for 64 mostly idle lanes.
  if (header.pad0 == 1u) { return; }
  let origin = cellCoord(header.cell);
  let size = header.size;
  var neighborCells: array<u32, 24>;
  var neighborCoefficients: array<f32, 24>;
  var neighborCount = 0u;
  var diagonal = 0.0;
  var flux = 0.0;
  for (var face = 0u; face < 6u; face += 1u) {
    let axis = face / 2u;
    let side = select(-1, 1, (face & 1u) == 1u);
    let e = axisVector(axis); let area = faceArea(axis);
    for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
      var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local); let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let neighbor = ownerAt(outside);
      let distance = pressureDistance(ownerAt(vec3i(origin)), neighbor, axis);
      let solid = faceSolid(inside, outside); let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let coefficient = open * area / max(distance, 1e-7);
      diagonal += coefficient;
      if (liquidOwner(neighbor)) {
        let neighborCell = index(unpackOrigin(neighbor.packedOrigin));
        var found = false;
        for (var j = 0u; j < neighborCount; j += 1u) {
          if (neighborCells[j] == neighborCell) { neighborCoefficients[j] += coefficient; found = true; break; }
        }
        if (!found && neighborCount < 24u) {
          neighborCells[neighborCount] = neighborCell;
          neighborCoefficients[neighborCount] = coefficient;
          neighborCount += 1u;
        }
      }
      let faceCell = select(outside, inside, side > 0);
      flux += f32(side) * area * constrainedFaceVelocity(faceCell, axis, solid);
    } }
  }
  header.entryCount = neighborCount; header.diagonal = diagonal; header.rhs = flux;
  leafHeaders[row] = header;
  for (var j = 0u; j < neighborCount; j += 1u) { leafEntries[header.entryStart + j] = LeafEntry(neighborCells[j], neighborCoefficients[j]); }
}

// A balanced neighbor is no smaller than half this leaf, so each face quadrant
// touches at most one neighbor. Sixty-four lanes can therefore reduce all
// size^2 finest subfaces into four deterministic coefficients per face without
// atomics. Duplicate quadrant entries for a same-size/coarser neighbor are
// algebraically identical to the merged serial entry and retain a fixed layout.
var<workgroup> coarseDiagonalScratch: array<f32, 64>;
var<workgroup> coarseFluxScratch: array<f32, 64>;
var<workgroup> coarseCoefficientScratch: array<f32, 256>;
var<workgroup> coarseTaskEligible: atomic<u32>;

@compute @workgroup_size(64)
fn assembleCoarseSystem(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let task = wid.x;
  if (lid == 0u) { atomicStore(&coarseTaskEligible, select(0u, 1u, coarseTaskTile(task) == 0u)); }
  workgroupBarrier();
  if (workgroupUniformLoad(&coarseTaskEligible) == 0u) { return; }
  let row = coarseTaskRow(task);
  var header = leafHeaders[row];
  let origin = cellCoord(header.cell);
  let owner = owners[header.cell];
  let size = header.size;
  let half = size / 2u;
  var diagonal = 0.0;
  var flux = 0.0;

  for (var face = 0u; face < 6u; face += 1u) {
    let axis = face / 2u;
    let side = select(-1, 1, (face & 1u) == 1u);
    let e = axisVector(axis);
    let area = faceArea(axis);
    var laneDiagonal = 0.0;
    var laneFlux = 0.0;
    var laneCoefficients = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    let faceSamples = size * size;
    for (var sample = lid; sample < faceSamples; sample += 64u) {
      let a = sample % size;
      let b = sample / size;
      var local = vec3u(0u);
      local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a;
      local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local);
      let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let neighbor = ownerAt(outside);
      let solid = faceSolid(inside, outside);
      let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let coefficient = open * area / max(pressureDistance(owner, neighbor, axis), 1e-7);
      laneDiagonal += coefficient;
      if (liquidOwner(neighbor)) {
        let quadrant = select(0u, 1u, a >= half) + select(0u, 2u, b >= half);
        laneCoefficients[quadrant] += coefficient;
      }
      let faceCell = select(outside, inside, side > 0);
      laneFlux += f32(side) * area * constrainedFaceVelocity(faceCell, axis, solid);
    }
    coarseDiagonalScratch[lid] = laneDiagonal;
    coarseFluxScratch[lid] = laneFlux;
    for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
      coarseCoefficientScratch[quadrant * 64u + lid] = laneCoefficients[quadrant];
    }
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) {
        coarseDiagonalScratch[lid] += coarseDiagonalScratch[lid + stride];
        coarseFluxScratch[lid] += coarseFluxScratch[lid + stride];
        for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
          let slot = quadrant * 64u + lid;
          coarseCoefficientScratch[slot] += coarseCoefficientScratch[slot + stride];
        }
      }
    }
    workgroupBarrier();
    if (lid == 0u) {
      diagonal += coarseDiagonalScratch[0];
      flux += coarseFluxScratch[0];
      for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) {
        let a = select(0u, half, (quadrant & 1u) != 0u);
        let b = select(0u, half, (quadrant & 2u) != 0u);
        var local = vec3u(0u);
        local[axis] = select(0u, size - 1u, side > 0);
        local[(axis + 1u) % 3u] = a;
        local[(axis + 2u) % 3u] = b;
        let outside = vec3i(origin + local) + side * e;
        var neighborCell = header.cell;
        let coefficient = coarseCoefficientScratch[quadrant * 64u];
        if (coefficient > 0.0 && valid(outside)) {
          let neighbor = ownerAt(outside);
          if (liquidOwner(neighbor)) { neighborCell = index(unpackOrigin(neighbor.packedOrigin)); }
        }
        leafEntries[header.entryStart + face * 4u + quadrant] = LeafEntry(neighborCell, coefficient);
      }
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    header.entryCount = 24u;
    header.diagonal = diagonal;
    header.rhs = flux;
    leafHeaders[row] = header;
  }
}

@compute @workgroup_size(256)
fn iterateLeaves(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  let header = leafHeaders[row];
  var sum = 0.0;
  for (var j = 0u; j < header.entryCount; j += 1u) {
    let entry = leafEntries[header.entryStart + j];
    sum += entry.coefficient * pressureIn[entry.cell];
  }
  let coupling = leafBodyCoupling(cellCoord(header.cell), header.size); let old = pressureIn[header.cell]; let effectiveDiagonal = header.diagonal + coupling.y;
  let next = select(0.0, old + (sum - header.rhs - header.diagonal * old - coupling.x) / effectiveDiagonal, effectiveDiagonal > 1e-8);
  pressureOut[header.cell] = mix(old, next, params.cellRelax.w);
}

// Chebyshev semi-iteration applies an accelerated polynomial to D^-1 A. Each
// row keeps its previous correction and recurrence scalar in the otherwise
// unused header padding, so every iteration remains one row-parallel SpMV.
@compute @workgroup_size(256)
fn iterateChebyshev(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  var header = leafHeaders[row];
  var sum = 0.0;
  for (var j = 0u; j < header.entryCount; j += 1u) {
    let entry = leafEntries[header.entryStart + j];
    sum += entry.coefficient * pressureIn[entry.cell];
  }
  let old = pressureIn[header.cell];
  let residual = select(0.0, (sum - header.rhs - header.diagonal * old) / header.diagonal, header.diagonal > 1e-8);
  let lower = params.solve.y; let upper = params.solve.z;
  let theta = 0.5 * (upper + lower); let delta = 0.5 * (upper - lower); let sigma = theta / delta;
  let previousSearch = bitcast<f32>(header.pad0); let previousRho = bitcast<f32>(header.pad1);
  var rho = 1.0 / sigma;
  var search = residual / theta;
  if (previousRho > 0.0) {
    rho = 1.0 / (2.0 * sigma - previousRho);
    search = rho * previousRho * previousSearch + (2.0 * rho / delta) * residual;
  }
  pressureOut[header.cell] = old + search;
  header.pad0 = bitcast<u32>(search); header.pad1 = bitcast<u32>(rho);
  leafHeaders[row] = header;
}

fn leafPressureLoad(parity: u32, cell: u32) -> f32 {
  if (parity == 0u) { return pressureIn[cell]; }
  return pressureOut[cell];
}
fn leafPressureStore(parity: u32, cell: u32, value: f32) {
  if (parity == 0u) { pressureOut[cell] = value; } else { pressureIn[cell] = value; }
}

var<workgroup> reduceScalars: array<f32, 256>;
var<workgroup> rowCountShared: u32;
var<workgroup> convergedShared: u32;

// The whole weighted-Jacobi loop in ONE dispatch. Single-workgroup execution
// makes storageBarrier sufficient for ping-pong coherence; the loop break must
// come from workgroupUniformLoad so every barrier stays in uniform control
// flow. Warm-started from the persistent pressure buffer and exits early once
// the residual falls below params.solve.x relative to |b|^2.
@compute @workgroup_size(256)
fn solveLeaves(@builtin(local_invocation_id) lid3: vec3u) {
  let lid = lid3.x;
  if (lid == 0u) { rowCountShared = compaction[0]; convergedShared = 0u; }
  let n = workgroupUniformLoad(&rowCountShared);
  let cap = params.control.y;
  let relaxation = params.cellRelax.w;
  let tolerance2 = params.solve.x;
  var normB = 1e-30;
  var parity = 0u;
  var executed = 0u;
  for (var it = 0u; it < cap; it += 1u) {
    if (workgroupUniformLoad(&convergedShared) != 0u) { break; }
    var rr = 0.0;
    var bb = 0.0;
    for (var row = lid; row < n; row += 256u) {
      let header = leafHeaders[row];
      var sum = 0.0;
      for (var j = 0u; j < header.entryCount; j += 1u) {
        let entry = leafEntries[header.entryStart + j];
        sum += entry.coefficient * leafPressureLoad(parity, entry.cell);
      }
      let previous = leafPressureLoad(parity, header.cell);
      let rowLive = header.diagonal > 1e-8;
      let next = select(0.0, (sum - header.rhs) / header.diagonal, rowLive);
      leafPressureStore(parity, header.cell, mix(previous, next, relaxation));
      let residual = select(0.0, sum - header.rhs - header.diagonal * previous, rowLive);
      rr += residual * residual;
      bb += header.rhs * header.rhs;
    }
    storageBarrier();
    if (it == 0u) {
      reduceScalars[lid] = bb;
      for (var stride = 128u; stride > 0u; stride >>= 1u) {
        workgroupBarrier();
        if (lid < stride) { reduceScalars[lid] += reduceScalars[lid + stride]; }
      }
      workgroupBarrier();
      normB = max(reduceScalars[0], 1e-30);
      workgroupBarrier();
    }
    reduceScalars[lid] = rr;
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) { reduceScalars[lid] += reduceScalars[lid + stride]; }
    }
    workgroupBarrier();
    if (lid == 0u && reduceScalars[0] <= tolerance2 * normB) { convergedShared = 1u; }
    parity ^= 1u;
    executed += 1u;
  }
  // Downstream stages always read pressureIn; fold an odd final parity back.
  storageBarrier();
  workgroupBarrier();
  if ((executed & 1u) == 1u) {
    for (var row = lid; row < n; row += 256u) {
      let cell = leafHeaders[row].cell;
      pressureIn[cell] = pressureOut[cell];
    }
  }
}

fn reconstructedAxisGradient(owner: Owner, axis: u32) -> f32 {
  let origin = unpackOrigin(owner.packedOrigin); let size = owner.size;
  let centrePressure = pressureOf(owner); let e = axisVector(axis);
  var gradientSum = 0.0; var weightSum = 0.0;
  for (var sideIndex = 0u; sideIndex < 2u; sideIndex += 1u) {
    let side = select(-1, 1, sideIndex == 1u);
    for (var b = 0u; b < size; b += 1u) { for (var a = 0u; a < size; a += 1u) {
      var local = vec3u(0u); local[axis] = select(0u, size - 1u, side > 0);
      local[(axis + 1u) % 3u] = a; local[(axis + 2u) % 3u] = b;
      let inside = vec3i(origin + local); let outside = inside + side * e;
      if (!valid(outside)) { continue; }
      let neighbor = ownerAt(outside);
      if (neighbor.packedOrigin == owner.packedOrigin) { continue; }
      let solid = faceSolid(inside, outside); let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let neighborPressure = select(0.0, pressureOf(neighbor), liquidOwner(neighbor));
      let gradient = f32(side) * (neighborPressure - centrePressure) / max(pressureDistance(owner, neighbor, axis), 1e-7);
      gradientSum += open * gradient; weightSum += open;
    } }
  }
  return select(0.0, gradientSum / weightSum, weightSum > 1e-6);
}

// Coarse-leaf affine reconstruction is surface-area work. Running the size^2
// face traversal in the single invocation at the leaf origin under-fills the
// GPU precisely for the largest leaves, where there are few origins and each
// invocation has thousands of subfaces. The 4^3 workgroup containing an
// aligned size>=4 origin owns that leaf and reduces its boundary cooperatively.
// vec3 components keep the three axis reductions independent while requiring
// only one workgroup reduction and one storage publication per leaf.
var<workgroup> gradientPartials: array<vec3f, 64>;
var<workgroup> gradientWeightPartials: array<vec3f, 64>;
var<workgroup> gradientOwnerOrigin: atomic<u32>;
var<workgroup> gradientOwnerSize: atomic<u32>;
var<workgroup> gradientOwnerEligible: atomic<u32>;

fn coarseGradientContribution(owner: Owner, sample: u32) -> array<vec3f, 2> {
  let origin = unpackOrigin(owner.packedOrigin);
  let faceSamples = owner.size * owner.size;
  let face = sample / faceSamples;
  let axis = face / 2u;
  let side = select(-1, 1, (face & 1u) == 1u);
  let inFace = sample - face * faceSamples;
  let a = inFace % owner.size;
  let b = inFace / owner.size;
  var local = vec3u(0u);
  local[axis] = select(0u, owner.size - 1u, side > 0);
  local[(axis + 1u) % 3u] = a;
  local[(axis + 2u) % 3u] = b;
  let inside = vec3i(origin + local);
  let outside = inside + side * axisVector(axis);
  var weighted = vec3f(0.0);
  var weight = vec3f(0.0);
  if (valid(outside)) {
    let neighbor = ownerAt(outside);
    if (neighbor.packedOrigin != owner.packedOrigin) {
      let solid = faceSolid(inside, outside);
      let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
      let neighborPressure = select(0.0, pressureOf(neighbor), liquidOwner(neighbor));
      let gradient = f32(side) * (neighborPressure - pressureOf(owner))
        / max(pressureDistance(owner, neighbor, axis), 1e-7);
      weighted[axis] = open * gradient;
      weight[axis] = open;
    }
  }
  return array<vec3f, 2>(weighted, weight);
}

fn reconstructedGradient(owner: Owner, axis: u32) -> f32 {
  if (owner.size <= 1u) { return 0.0; }
  var slot = unpackOrigin(owner.packedOrigin); slot[axis] += 1u;
  return pressureOut[index(slot)];
}

@compute @workgroup_size(64)
fn reconstructGradients(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let task = wid.x;
  let row = coarseTaskRow(task);
  if (lid == 0u) {
    var packed = 0u; var size = 0u;
    let eligible = row < compaction[0] && coarseTaskTile(task) == 0u;
    if (eligible) { let header = leafHeaders[row]; packed = owners[header.cell].packedOrigin; size = header.size; }
    atomicStore(&gradientOwnerOrigin, packed);
    atomicStore(&gradientOwnerSize, size);
    atomicStore(&gradientOwnerEligible, select(0u, 1u, eligible));
  }
  workgroupBarrier();
  let coarseOwner = Owner(workgroupUniformLoad(&gradientOwnerOrigin), workgroupUniformLoad(&gradientOwnerSize));
  if (workgroupUniformLoad(&gradientOwnerEligible) == 0u || coarseOwner.size <= 1u) { return; }
  let workgroupOrigin = unpackOrigin(coarseOwner.packedOrigin);
  if (coarseOwner.size >= 4u) {
    var weighted = vec3f(0.0);
    var weight = vec3f(0.0);
    let sampleCount = 6u * coarseOwner.size * coarseOwner.size;
    for (var sample = lid; sample < sampleCount; sample += 64u) {
      let contribution = coarseGradientContribution(coarseOwner, sample);
      weighted += contribution[0];
      weight += contribution[1];
    }
    gradientPartials[lid] = weighted;
    gradientWeightPartials[lid] = weight;
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      workgroupBarrier();
      if (lid < stride) {
        gradientPartials[lid] += gradientPartials[lid + stride];
        gradientWeightPartials[lid] += gradientWeightPartials[lid + stride];
      }
    }
    workgroupBarrier();
    if (lid == 0u) {
      let gradient = vec3f(
        select(0.0, gradientPartials[0].x / gradientWeightPartials[0].x, gradientWeightPartials[0].x > 1e-6),
        select(0.0, gradientPartials[0].y / gradientWeightPartials[0].y, gradientWeightPartials[0].y > 1e-6),
        select(0.0, gradientPartials[0].z / gradientWeightPartials[0].z, gradientWeightPartials[0].z > 1e-6)
      );
      // These slots are strictly inside a size>=2 leaf and therefore cannot
      // be pressure origins. pressureOut is the non-final ping-pong buffer.
      pressureOut[index(workgroupOrigin + vec3u(1u,0u,0u))] = gradient.x;
      pressureOut[index(workgroupOrigin + vec3u(0u,1u,0u))] = gradient.y;
      pressureOut[index(workgroupOrigin + vec3u(0u,0u,1u))] = gradient.z;
    }
    return;
  }
  if (lid == 0u) {
    pressureOut[index(workgroupOrigin + vec3u(1u,0u,0u))] = reconstructedAxisGradient(coarseOwner, 0u);
    pressureOut[index(workgroupOrigin + vec3u(0u,1u,0u))] = reconstructedAxisGradient(coarseOwner, 1u);
    pressureOut[index(workgroupOrigin + vec3u(0u,0u,1u))] = reconstructedAxisGradient(coarseOwner, 2u);
  }
}

@compute @workgroup_size(256)
fn reconstructSmallGradients(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  let header = leafHeaders[row];
  // Chebyshev reuses pad0/pad1 as spectral state after assembly. Classify
  // projection work from the immutable leaf size, never those scratch words.
  if (header.size >= 8u || header.size <= 1u) { return; }
  let owner = owners[header.cell];
  let origin = unpackOrigin(owner.packedOrigin);
  pressureOut[index(origin + vec3u(1u,0u,0u))] = reconstructedAxisGradient(owner, 0u);
  pressureOut[index(origin + vec3u(0u,1u,0u))] = reconstructedAxisGradient(owner, 1u);
  pressureOut[index(origin + vec3u(0u,0u,1u))] = reconstructedAxisGradient(owner, 2u);
}

fn projectedComponent(id: vec3i, axis: u32, input: f32) -> f32 {
  let neighborCell = id + axisVector(axis); let d = vec3i(dims());
  if (id[axis] == d[axis] - 1) { return 0.0; }
  let left = ownerAt(id); let right = ownerAt(neighborCell); let leftWet = liquidOwner(left); let rightWet = liquidOwner(right); let solid = faceSolid(id, neighborCell);
  var fluid = input;
  if (leftWet || rightWet) {
    if (left.packedOrigin != right.packedOrigin) {
      let p0 = select(0.0, pressureOf(left), leftWet); let p1 = select(0.0, pressureOf(right), rightWet);
      let distance = pressureDistance(left, right, axis);
      fluid -= (p1 - p0) / max(distance, 1e-7);
    } else if (leftWet && rightWet) {
      fluid -= reconstructedGradient(left, axis);
    }
  } else { fluid = 0.0; }
  let open = 1.0 - clamp(solid.fraction, 0.0, 1.0);
  let constrained = open * fluid + solid.fraction * component(solidVelocity(solid, faceWorld(id, axis)), axis);
  return clamp(constrained, -50.0, 50.0);
}

fn liveOwner(owner: Owner) -> bool { return frontierAlive(index(unpackOrigin(owner.packedOrigin))); }
fn projectionController(id: vec3i) -> u32 {
  if (!valid(id)) { return 0xffffffffu; }
  var best = 0xffffffffu;
  let candidates = array<vec3i,4>(id, id + vec3i(1,0,0), id + vec3i(0,1,0), id + vec3i(0,0,1));
  for (var candidate = 0u; candidate < 4u; candidate += 1u) {
    let p = candidates[candidate];
    if (!valid(p)) { continue; }
    let owner = ownerAt(p);
    if (liveOwner(owner)) { best = min(best, owner.packedOrigin); }
  }
  return best;
}

@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let id = vec3i(gid); let input = textureLoad(velocityIn, id, 0).xyz;
  let v = vec3f(projectedComponent(id,0u,input.x), projectedComponent(id,1u,input.y), projectedComponent(id,2u,input.z));
  let known = select(0.0, 1.0, liquidCell(id) || liquidCell(id+vec3i(1,0,0)) || liquidCell(id+vec3i(0,1,0)) || liquidCell(id+vec3i(0,0,1)));
  textureStore(velocityOut, id, vec4f(v, known));
}

fn projectLeafCell(owner: Owner, id: vec3i) {
  if (!valid(id) || projectionController(id) != owner.packedOrigin) { return; }
  let input = textureLoad(velocityIn, id, 0).xyz;
  let v = vec3f(projectedComponent(id,0u,input.x), projectedComponent(id,1u,input.y), projectedComponent(id,2u,input.z));
  textureStore(velocityOut, id, vec4f(v, f32(frontierGeneration())));
}

// Small leaves retain row-parallel occupancy: one invocation owns a row and
// performs at most 5^3 compatibility writes. Overflow coarse rows use this
// path too, so the cooperative-list capacity never affects correctness.
@compute @workgroup_size(256)
fn projectSmallLeaves(@builtin(global_invocation_id) gid: vec3u) {
  let row = compactRowIndex(gid);
  if (row >= compaction[0]) { return; }
  let header = leafHeaders[row];
  if (header.size >= 8u) { return; }
  let owner = owners[header.cell];
  let width = owner.size + 1u;
  let origin = vec3i(unpackOrigin(owner.packedOrigin));
  for (var sample = 0u; sample < width * width * width; sample += 1u) {
    let local = vec3i(i32(sample % width), i32((sample / width) % width), i32(sample / (width * width)));
    projectLeafCell(owner, origin + local - vec3i(1));
  }
}

// One cooperative workgroup owns each 8^3 tile of a size-8+ leaf. Boundary
// tiles also publish the leaf's negative-face halo. At most 9^3 samples are
// spread over 256 lanes, while a 32^3 leaf exposes 64 independent workgroups.
@compute @workgroup_size(256)
fn projectLeaves(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let task = wid.x;
  let row = coarseTaskRow(task);
  let tile = coarseTaskTile(task);
  let header = leafHeaders[row];
  let owner = owners[header.cell];
  let tiles = owner.size / 8u;
  let tileCoord = vec3u(tile % tiles, (tile / tiles) % tiles, tile / (tiles * tiles));
  let halo = vec3u(select(0u, 1u, tileCoord.x == 0u), select(0u, 1u, tileCoord.y == 0u), select(0u, 1u, tileCoord.z == 0u));
  let extent = vec3u(8u) + halo;
  let samples = extent.x * extent.y * extent.z;
  let tileOrigin = vec3i(unpackOrigin(owner.packedOrigin) + tileCoord * 8u) - vec3i(halo);
  for (var sample = lid; sample < samples; sample += 256u) {
    let local = vec3i(i32(sample % extent.x), i32((sample / extent.x) % extent.y), i32(sample / (extent.x * extent.y)));
    projectLeafCell(owner, tileOrigin + local);
  }
}

// The sparse projector deliberately leaves unknown air texels untouched. The
// projector stamps known cells with the current frontier generation. The first
// extrapolation sweep ignores stale alpha from prior frames, materializes a
// complete dense compatibility texture, and expands the band by one cell.
@compute @workgroup_size(4,4,4)
fn extrapolateSeed(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let id = vec3i(gid);
  let center = textureLoad(velocityIn, id, 0);
  let generation = frontierGeneration();
  if (u32(round(center.w)) == generation) { textureStore(velocityOut, id, vec4f(center.xyz, 1.0)); return; }
  let offsets = array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var sum = vec3f(0.0); var weight = 0.0;
  for (var n = 0u; n < 6u; n += 1u) {
    let p = id + offsets[n];
    if (valid(p)) { let q = textureLoad(velocityIn, p, 0); if (u32(round(q.w)) == generation) { sum += q.xyz; weight += 1.0; } }
  }
  textureStore(velocityOut, id, select(vec4f(0.0), vec4f(sum / max(weight,1.0),1.0), weight > 0.0));
}

@compute @workgroup_size(4,4,4)
fn extrapolate(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let id = vec3i(gid); let center = textureLoad(velocityIn, id, 0);
  if (center.w > 0.5) { textureStore(velocityOut, id, center); return; }
  var sum = vec3f(0.0); var weight = 0.0;
  let offsets = array<vec3i,6>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1));
  for (var n = 0u; n < 6u; n += 1u) { let p = id + offsets[n]; if (valid(p)) { let q = textureLoad(velocityIn,p,0); if (q.w > 0.5) { sum += q.xyz; weight += 1.0; } } }
  textureStore(velocityOut, id, select(vec4f(0.0), vec4f(sum / max(weight,1.0),1.0), weight > 0.0));
}
`;

/** Pressure part of the octree VOS coupling, accumulated into the shared
 * fixed-point body exchange alongside the post-projection tangential blend. */
export const octreePressureCouplingShader = /* wgsl */ `
struct Owner { packedOrigin: u32, size: u32 }
struct SolidCell { fraction: f32, owner: i32 }
struct RigidBody { positionShape: vec4f, dimensions: vec4f, orientation: vec4f, linearVelocity: vec4f, angularVelocity: vec4f, inverseMassInertia: vec4f, angularMomentumRestitution: vec4f, material: vec4f }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f }
@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> owners: array<Owner>;
@group(0) @binding(2) var<storage, read> solidCells: array<SolidCell>;
@group(0) @binding(3) var<storage, read> rigidBodies: array<RigidBody, 12>;
@group(0) @binding(4) var<storage, read_write> rigidExchange: array<atomic<i32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var levelSetIn: texture_3d<f32>;
fn dims() -> vec3u { return params.dimsMax.xyz; }
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn unpackOrigin(word: u32) -> vec3u { return vec3u(word & 1023u, (word >> 10u) & 1023u, (word >> 20u) & 1023u); }
fn ownerAt(p: vec3i) -> Owner { return owners[index(vec3u(p))]; }
fn phi(p: vec3i) -> f32 { return textureLoad(levelSetIn, p, 0).x; }
fn ownerPhi(owner: Owner) -> f32 {
  let centre = vec3f(unpackOrigin(owner.packedOrigin)) + vec3f(0.5 * f32(owner.size - 1u));
  let a = vec3u(floor(centre)); let b = min(a + vec3u(1u), dims() - vec3u(1u)); let t = fract(centre);
  let p000 = phi(vec3i(a)); let p100 = phi(vec3i(vec3u(b.x,a.y,a.z)));
  let p010 = phi(vec3i(vec3u(a.x,b.y,a.z))); let p110 = phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001 = phi(vec3i(vec3u(a.x,a.y,b.z))); let p101 = phi(vec3i(vec3u(b.x,a.y,b.z)));
  let p011 = phi(vec3i(vec3u(a.x,b.y,b.z))); let p111 = phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn liquidOwner(owner: Owner) -> bool { return ownerPhi(owner) < 0.0; }
fn pressureOf(owner: Owner) -> f32 { return pressure[index(unpackOrigin(owner.packedOrigin))]; }
fn axisVector(axis: u32) -> vec3i { return select(select(vec3i(0,0,1), vec3i(0,1,0), axis == 1u), vec3i(1,0,0), axis == 0u); }
fn faceArea(axis: u32) -> f32 { let h = params.cellRelax.xyz; return select(select(h.x * h.y, h.x * h.z, axis == 1u), h.y * h.z, axis == 0u); }
fn worldCell(p: vec3i) -> vec3f { let h = params.cellRelax.xyz; return vec3f(-0.5 * params.container.x + (f32(p.x) + 0.5) * h.x, (f32(p.y) + 0.5) * h.y, -0.5 * params.container.z + (f32(p.z) + 0.5) * h.z); }
fn faceWorld(p: vec3i, axis: u32) -> vec3f { var result = worldCell(p); result[axis] += 0.5 * params.cellRelax[axis]; return result; }
fn faceSolid(a: vec3i, b: vec3i) -> SolidCell { let sa = solidCells[index(vec3u(a))]; let sb = solidCells[index(vec3u(b))]; if (sa.fraction >= sb.fraction) { return sa; } return sb; }
fn addImpulse(bodyIndex: u32, impulse: vec3f, world: vec3f) {
  let body = rigidBodies[bodyIndex]; let torque = cross(world - body.positionShape.xyz, impulse); let base = bodyIndex * 12u;
  atomicAdd(&rigidExchange[base], i32(round(impulse.x * 1e6))); atomicAdd(&rigidExchange[base + 1u], i32(round(impulse.y * 1e6))); atomicAdd(&rigidExchange[base + 2u], i32(round(impulse.z * 1e6)));
  atomicAdd(&rigidExchange[base + 3u], i32(round(torque.x * 1e6))); atomicAdd(&rigidExchange[base + 4u], i32(round(torque.y * 1e6))); atomicAdd(&rigidExchange[base + 5u], i32(round(torque.z * 1e6)));
}
@compute @workgroup_size(4,4,4)
fn accumulatePressureImpulse(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let p = vec3i(gid);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let q = p + axisVector(axis); if (!valid(q)) { continue; }
    let solid = faceSolid(p, q); if (solid.owner < 0 || solid.fraction <= 0.0 || u32(solid.owner) >= params.control.w) { continue; }
    let left = ownerAt(p); let right = ownerAt(q); let leftWet = liquidOwner(left); let rightWet = liquidOwner(right);
    if (!leftWet && !rightWet) { continue; }
    let p0 = select(0.0, pressureOf(left), leftWet); let p1 = select(0.0, pressureOf(right), rightWet);
    let scalar = params.physical.x * faceArea(axis) * solid.fraction * (p0 - p1);
    var impulse = vec3f(0.0); impulse[axis] = scalar;
    addImpulse(u32(solid.owner), impulse, faceWorld(p, axis));
  }
}
`;

/** GPU-only adapter from the dense owner map to the scientific overlay fields. */
export const octreeDiagnosticShader = /* wgsl */ `
struct Owner { packedOrigin: u32, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f }
@group(0) @binding(0) var<storage, read> owners: array<Owner>;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var velocity: texture_3d<f32>;
@group(0) @binding(3) var levelSetIn: texture_3d<f32>;
@group(0) @binding(4) var topologyOut: texture_storage_3d<rg32uint, write>;
@group(0) @binding(5) var pressureSamplesOut: texture_storage_3d<rgba32uint, write>;
@group(0) @binding(6) var pressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var divergenceOut: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var<uniform> params: Params;
@group(0) @binding(9) var<storage, read> gradients: array<f32>;
@group(0) @binding(10) var velocityBeforeProjection: texture_3d<f32>;
fn dims() -> vec3u { return params.dimsMax.xyz; }
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn unpackOrigin(word: u32) -> vec3u { return vec3u(word & 1023u, (word >> 10u) & 1023u, (word >> 20u) & 1023u); }
fn phi(p: vec3i) -> f32 { return textureLoad(levelSetIn, p, 0).x; }
fn liquidCell(p: vec3i) -> bool { return valid(p) && phi(p) < 0.0; }
fn ownerPhi(owner: Owner) -> f32 {
  let centre = vec3f(unpackOrigin(owner.packedOrigin)) + vec3f(0.5 * f32(owner.size - 1u));
  let a = vec3u(floor(centre)); let b = min(a + vec3u(1u), dims() - vec3u(1u)); let t = fract(centre);
  let p000 = phi(vec3i(a)); let p100 = phi(vec3i(vec3u(b.x,a.y,a.z)));
  let p010 = phi(vec3i(vec3u(a.x,b.y,a.z))); let p110 = phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001 = phi(vec3i(vec3u(a.x,a.y,b.z))); let p101 = phi(vec3i(vec3u(b.x,a.y,b.z)));
  let p011 = phi(vec3i(vec3u(a.x,b.y,b.z))); let p111 = phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y), mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y), t.z);
}
fn liquidOwner(owner: Owner) -> bool {
  return ownerPhi(owner) < 0.0;
}
fn leafGradient(owner: Owner) -> vec3f {
  if (owner.size <= 1u) { return vec3f(0.0); }
  let origin = unpackOrigin(owner.packedOrigin);
  return vec3f(gradients[index(origin + vec3u(1u,0u,0u))], gradients[index(origin + vec3u(0u,1u,0u))], gradients[index(origin + vec3u(0u,0u,1u))]);
}
fn velocityAt(p: vec3i) -> vec3f { if (!valid(p)) { return vec3f(0.0); } return textureLoad(velocity, p, 0).xyz; }
@compute @workgroup_size(4,4,4)
fn materializeOctreeFields(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = owners[index(gid)]; let origin = unpackOrigin(owner.packedOrigin); let ownerIndex = index(origin);
  let horizontal = origin.x | (origin.z << 10u) | (owner.size << 20u);
  let vertical = origin.y | ((origin.y + owner.size) << 10u);
  textureStore(topologyOut, vec3i(gid), vec4u(horizontal, vertical, 0u, 0u));
  let wet = liquidOwner(owner); let invalid = 0xffffffffu;
  let q = vec3i(gid);
  let pressureUpdate = length(velocityAt(q) - textureLoad(velocityBeforeProjection, q, 0).xyz);
  // Octree ownership needs only x/z/w. The otherwise-unused y lane carries a
  // bitcast live scalar for the Projection Δu overlay without another texture.
  textureStore(pressureSamplesOut, q, select(vec4u(invalid), vec4u(ownerIndex, bitcast<u32>(pressureUpdate), vertical, horizontal), wet));
  let centre = vec3f(origin) + vec3f(0.5 * f32(owner.size - 1u));
  let offset = (vec3f(gid) - centre) * params.cellRelax.xyz;
  let mappedPressure = pressure[ownerIndex] + dot(leafGradient(owner), offset);
  textureStore(pressureOut, vec3i(gid), vec4f(select(0.0, mappedPressure, wet)));
  let h = params.cellRelax.xyz;
  let divergence = (velocityAt(q).x - velocityAt(q-vec3i(1,0,0)).x) / h.x
                 + (velocityAt(q).y - velocityAt(q-vec3i(0,1,0)).y) / h.y
                 + (velocityAt(q).z - velocityAt(q-vec3i(0,0,1)).z) / h.z;
  textureStore(divergenceOut, q, vec4f(select(0.0, divergence, wet)));
}
`;
