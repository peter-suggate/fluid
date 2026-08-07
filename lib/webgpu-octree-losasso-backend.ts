import type { OctreeOwnerPagePlan } from "./webgpu-octree-owner-pages";
import type { PassBroker } from "./webgpu-pass-broker";
import type { LosassoFreeSurfacePressureMode, LosassoVelocityExtensionMode } from
  "./octree-coarse-backend";
import type { OctreeFirstOrderSPDVCycle } from "./webgpu-octree-section43-contract";
import type { OctreeLosassoSolveTuning } from "./octree-runtime-dials";
import {
  WebGPUOctreeLosassoAuthority,
  type WebGPUOctreeLosassoAuthorityCapacities,
  type WebGPUOctreeLosassoWritableAuthority,
} from "./webgpu-octree-losasso-authority";
import { webgpuOctreeLosassoBackendWGSL } from "./webgpu-octree-losasso-backend.wgsl";
import { webgpuOctreeLosassoAuthorityCommitWGSL } from
  "./webgpu-octree-losasso-authority-commit.wgsl";
import {
  assertOctreeLosassoVCycle,
  type OctreeLosassoFirstOrderVCycle,
  type OctreeLosassoVCycleHierarchySource,
  type OctreeLosassoVCycleLevelSource,
  type OctreeLosassoVCycleTransferSource,
} from "./webgpu-octree-losasso-vcycle";
import { WebGPUOctreeLosassoVCycle } from "./webgpu-octree-losasso-vcycle-gpu";
import { WebGPUOctreeLosassoHierarchyPublisher } from
  "./webgpu-octree-losasso-hierarchy";
import {
  WebGPUOctreeLosassoOperator,
  type WebGPUOctreeLosassoOperatorSource,
} from "./webgpu-octree-losasso-operator";
import {
  WebGPUOctreeLosassoProjection,
  type WebGPUOctreeLosassoProjectionSource,
} from "./webgpu-octree-losasso-projection";
import type { WebGPUOctreeLosassoVelocityExtensionSource } from
  "./webgpu-octree-losasso-velocity-extension";
import {
  WebGPUOctreeLosassoExtensionBand,
} from "./webgpu-octree-losasso-extension-band";
import { WebGPUOctreeLosassoVelocityMigration } from
  "./webgpu-octree-losasso-velocity-migration";
import type { WebGPUFineLevelSetBrickSource } from
  "./webgpu-octree-fine-levelset-bricks";
import type { WebGPUOctreeLosassoCoarsePhiSource } from
  "./webgpu-octree-losasso-coarse-phi";
import type { WebGPUOctreeLosassoVelocitySamplerSource } from
  "./webgpu-octree-losasso-velocity-sampler";
import {
  WebGPUOctreeLosassoDynamics,
  type WebGPUOctreeLosassoDynamicsSource,
  type WebGPUOctreeLosassoDynamicsStep,
} from "./webgpu-octree-losasso-dynamics";
import { WebGPUOctreeLosassoRigidPressureReaction } from
  "./webgpu-octree-losasso-rigid-pressure-reaction";
import {
  WebGPUOctreePipelinedMGPCG,
  type OctreePipelinedMGPCGOptions,
  type OctreePipelinedMGPCGVectors,
} from "./webgpu-octree-pipelined-mgpcg";
import { WebGPUOctreeLosassoResidentMGPCG } from
  "./webgpu-octree-losasso-resident-mgpcg";

export interface LosassoInitializationTask {
  readonly label: string;
  readonly run: () => Promise<void>;
}

/** The complete recurring topology input. No Power-only source can be passed. */
export interface WebGPUOctreeLosassoCandidateInput {
  /** Existing 48-byte compact pressure LeafHeader rows. */
  readonly leafHeaders: GPUBuffer;
  /** Finalized dual-bank frontier; words 7/0/1 select the candidate row count. */
  readonly frontier: GPUBuffer;
  /** Dual-bank owner arena; publication reads the inactive candidate table. */
  readonly ownerArena: GPUBuffer;
  /** Owner-page transaction whose inactive table matches candidate rows. */
  readonly ownerCandidateTransaction: GPUBuffer;
  /** Dense two-word `{solidFraction, rigidOwner}` records on finest cells. */
  readonly solidCells: GPUBuffer;
  /** Shared scene rigid-body state, using the coarse projection's 32-f32 ABI. */
  readonly rigidBodies: GPUBuffer;
}

export interface WebGPUOctreeLosassoPublishedSources {
  readonly operator: WebGPUOctreeLosassoOperatorSource;
  readonly projection: WebGPUOctreeLosassoProjectionSource;
  readonly dynamics: WebGPUOctreeLosassoDynamicsSource;
  readonly extension: WebGPUOctreeLosassoVelocityExtensionSource;
  readonly vcycle: OctreeLosassoVCycleHierarchySource;
  readonly rightHandSide: GPUBuffer;
  readonly rowCount: GPUBuffer;
  /** Present on the constructible backend's owned authority. */
  readonly velocitySampler?: WebGPUOctreeLosassoVelocitySamplerSource;
  readonly wideSolver?: Readonly<{
    diagonal: GPUBuffer;
    acceptedAuthority: GPUBuffer;
  }>;
}

export interface WebGPUOctreeLosassoCandidatePublisher {
  readonly initializationTasks?: readonly LosassoInitializationTask[];
  readonly sources: WebGPUOctreeLosassoPublishedSources;
  readonly allocatedBytes?: number;
  encodeCandidatePublication(broker: PassBroker, input: WebGPUOctreeLosassoCandidateInput): void;
  destroy(): void;
}

export interface WebGPUOctreeLosassoPressureSolver {
  readonly initializationTasks?: readonly LosassoInitializationTask[];
  readonly allocatedBytes?: number;
  readonly control?: GPUBuffer;
  readonly iterationBudget?: number;
  readonly symmetryStageAuditBuffers?: WebGPUOctreePipelinedMGPCG["symmetryStageAuditBuffers"];
  encodeSolve(broker: PassBroker, input: {
    readonly pressureSeed: GPUBuffer;
    readonly pressureOut: GPUBuffer;
    readonly rightHandSide: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
  /** Adopt live accuracy/cost dials. Absent on solvers a harness substitutes. */
  setSolveTuning?(tuning: OctreeLosassoSolveTuning): void;
  destroy(): void;
}

export interface WebGPUOctreeLosassoTopologyPlan {
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  readonly physicalCellSize: readonly [number, number, number];
  readonly domainOrigin?: readonly [number, number, number];
  /** The plan belonging to `candidate.ownerArena`; its immutable offsets form
   * the owner lookup ABI and do not load any Power descriptor state. */
  readonly ownerPages: Pick<OctreeOwnerPagePlan,
    "brickDimensions" | "logicalBrickCount" | "ownerDirectoryOffsetWords"
    | "ownerPagesOffsetWords" | "capacity" | "pageVoxels">;
}

export interface WebGPUOctreeLosassoBackendOptions {
  readonly device: GPUDevice;
  readonly capacities: WebGPUOctreeLosassoAuthorityCapacities;
  readonly topology: WebGPUOctreeLosassoTopologyPlan;
  readonly density: number;
  /** Compact air-face headroom: six velocity-cell faces per resident B4 page. */
  readonly extensionBandBrickCapacity: number;
  readonly freeSurfacePressureMode?: LosassoFreeSurfacePressureMode;
  readonly velocityExtensionMode?: LosassoVelocityExtensionMode;
  readonly closedBoundaries?: readonly [boolean, boolean, boolean, boolean, boolean, boolean];
  /** Coarser levels use exactly the same first-order axis-face operator. */
  readonly coarseLevels?: readonly OctreeLosassoVCycleLevelSource[];
  readonly transfers?: readonly OctreeLosassoVCycleTransferSource[];
  readonly createVCycle?: (source: OctreeLosassoVCycleHierarchySource) => OctreeFirstOrderSPDVCycle;
  readonly createSolver?: (input: {
    readonly operator: WebGPUOctreeLosassoOperator;
    readonly preconditioner: OctreeLosassoFirstOrderVCycle;
  }) => WebGPUOctreeLosassoPressureSolver;
  /**
   * Select the single-dispatch resident MGPCG executor for the ≤4K-row tier.
   * Falls back to the wide pipelined solver when the hierarchy or device
   * cannot host the resident kernel.
   */
  readonly residentSolver?: boolean;
  readonly solver?: Omit<OctreePipelinedMGPCGOptions, "rowCapacity">;
  /** Shared rigid exchange seam. Kept optional for standalone reduced backends. */
  readonly rigidPressureReaction?: Readonly<{
    solidCells: GPUBuffer;
    rigidBodies: GPUBuffer;
    rigidExchange: GPUBuffer;
    /** Finest-cell lattice origin in the centred rigid-body world. */
    rigidWorldOrigin: readonly [number, number, number];
    /** Authored hydrostatic gauge plane used to isolate dynamic pressure. */
    hydrostaticReferenceY_m: number;
  }>;
}

const PUBLISH_ENTRY_POINTS = [
  "beginLosassoPublication", "countLosassoFaces", "prefixLosassoFaces",
  "clearLosassoFaceDirectory",
  "emitLosassoFaces", "conditionLosassoFaces", "prefixLosassoIncidences", "scatterLosassoIncidences",
  "sortLosassoIncidences", "clearLosassoAdjacencyOffsets", "buildLosassoFaceDirectory",
  "finishLosassoPublication",
] as const;
const AUTHORITY_COMMIT_ENTRY_POINTS = [
  "commitLosassoAuthorityRows", "commitLosassoAuthorityIncidences",
  "commitLosassoAuthorityFaces", "commitLosassoAuthorityFaceAux",
  "commitLosassoAuthorityVelocity", "commitLosassoAuthorityDirectory",
  "finishLosassoAuthorityCommit",
] as const;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new RangeError(`Losasso ${label} must be a positive u32`);
  }
  return value;
}

function topologyParameterWords(plan: WebGPUOctreeLosassoTopologyPlan,
  capacities: Readonly<Required<WebGPUOctreeLosassoAuthorityCapacities>>,
  closedBoundaries?: readonly [boolean, boolean, boolean, boolean, boolean, boolean],
  rigidWorldOrigin?: readonly [number, number, number],
): ArrayBuffer {
  const [nx, ny, nz] = plan.dimensions;
  [nx, ny, nz].forEach((value, axis) => positiveInteger(value, `dimension ${axis}`));
  positiveInteger(plan.maximumLeafSize, "maximum leaf size");
  if ((plan.maximumLeafSize & (plan.maximumLeafSize - 1)) !== 0) {
    throw new RangeError("Losasso maximum leaf size must be dyadic");
  }
  for (const [axis, value] of plan.physicalCellSize.entries()) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new RangeError(`Losasso physical cell size ${axis} must be positive and finite`);
    }
  }
  const owner = plan.ownerPages;
  const words = new Uint32Array(28);
  words.set([nx, ny, nz, plan.maximumLeafSize], 0);
  words.set([...owner.brickDimensions, owner.logicalBrickCount], 4);
  words.set([owner.ownerDirectoryOffsetWords, owner.ownerPagesOffsetWords,
    owner.capacity, owner.pageVoxels], 8);
  // The directory is power-of-two and at most half full by construction.
  let directoryCapacity = 1;
  while (directoryCapacity < 2 * capacities.faces) directoryCapacity *= 2;
  words.set([capacities.rows, capacities.faces, capacities.incidences,
    directoryCapacity], 12);
  new Float32Array(words.buffer).set([...plan.physicalCellSize, 0], 16);
  // Topology itself is indexed in grid-local coordinates. The only world-space
  // consumer of this origin is analytic rigid conditioning, whose state buffer
  // lives in the centred rigid-body world.
  const origin = rigidWorldOrigin ?? plan.domainOrigin ?? [0, 0, 0];
  if (origin.some((value) => !Number.isFinite(value))) {
    throw new RangeError("Losasso domain origin must be finite");
  }
  new Float32Array(words.buffer).set([...origin, 0], 20);
  words[24] = (closedBoundaries ?? [true,true,true,true,true,true])
    .reduce((mask,value,index)=>mask|(value?1<<index:0),0)>>>0;
  return words.buffer;
}

/**
 * GPU-owned compact face publisher. Candidate construction is isolated from
 * the fixed accepted buffers consumed by the live solver and dynamics graph.
 */
class WebGPUOctreeLosassoTopologyPublisher implements WebGPUOctreeLosassoCandidatePublisher {
  readonly authority: WebGPUOctreeLosassoAuthority;
  readonly sources: WebGPUOctreeLosassoPublishedSources;
  readonly allocatedBytes: number;
  readonly publication = "candidate-ready-commit" as const;
  readonly extensionPublication = "all-projected-faces-are-seeds" as const;

  private readonly params: GPUBuffer;
  private readonly commitParams: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly hierarchyPublisher?: WebGPUOctreeLosassoHierarchyPublisher;
  private readonly velocityMigration: WebGPUOctreeLosassoVelocityMigration;
  private readonly bindGroupCache = new Map<GPUComputePipeline,
    { bindings: readonly number[]; buffers: readonly GPUBuffer[]; group: GPUBindGroup }[]>();
  /** Refresh-time validation lives here, never in the accepted authority. */
  private readonly acceptedRigidBoundaryControl: GPUBuffer;
  private acceptedRigidBoundaryGroup?: GPUBindGroup;
  private pipelines?: readonly GPUComputePipeline[];
  private commitPipelines?: readonly GPUComputePipeline[];
  private destroyed = false;

  get acceptedRigidBoundaryDiagnostics(): GPUBuffer {
    return this.acceptedRigidBoundaryControl;
  }

  constructor(private readonly device: GPUDevice,
    options: Pick<WebGPUOctreeLosassoBackendOptions,
      "capacities" | "topology" | "coarseLevels" | "transfers" | "closedBoundaries"
      | "rigidPressureReaction">) {
    // The geometric publisher has no non-seed extension graph yet. One valid
    // adjacency word keeps the reduced binding constructible without carrying
    // the old frontier arena.
    if ((options.coarseLevels === undefined) !== (options.transfers === undefined)) {
      throw new Error("Losasso coarse levels and transfers must be supplied together");
    }
    this.authority = new WebGPUOctreeLosassoAuthority(device, {
      capacities: { ...options.capacities, faceAdjacencies: 1 },
      coarseLevels: options.coarseLevels,
      transfers: options.transfers,
      encodeCandidate: (broker, input, output) => this.encode(broker, input, output),
    });
    if (!options.coarseLevels) {
      this.hierarchyPublisher = new WebGPUOctreeLosassoHierarchyPublisher(device, {
        rowCapacity: options.capacities.rows,
        faceCapacity: options.capacities.faces,
        dimensions: options.topology.dimensions,
        maximumLeafSize: options.topology.maximumLeafSize,
        physicalCellSize: options.topology.physicalCellSize,
        finest: this.authority.sources.vcycle.levels[0]!,
      });
    }
    this.velocityMigration = new WebGPUOctreeLosassoVelocityMigration(device, {
      dimensions: options.topology.dimensions,
      maximumLeafSize: options.topology.maximumLeafSize,
      faceCapacity: this.authority.writable.capacities.faces,
      directoryCapacity: this.authority.writable.faceDirectoryCapacity,
    });
    this.sources = Object.freeze({ ...this.authority.sources,
      vcycle: this.hierarchyPublisher?.hierarchy ?? this.authority.sources.vcycle,
      velocitySampler: {
        control: this.authority.writable.control,
        faceGeometry: this.authority.writable.faceGeometry,
        extendedVelocity: this.authority.writable.extendedVelocity,
        axisFaceDirectory: this.authority.writable.axisFaceDirectory,
        directoryCapacity: this.authority.writable.faceDirectoryCapacity,
        dimensions: options.topology.dimensions,
        maximumLeafSize: options.topology.maximumLeafSize,
        fineCellSize: options.topology.physicalCellSize[0],
      },
      wideSolver: {
        diagonal: this.authority.writable.diagonal,
        acceptedAuthority: this.authority.writable.solverAuthority,
      },
    });
    const capacities = this.authority.writable.capacities;
    this.params = device.createBuffer({ label: "Losasso compact topology parameters",
      size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0,
      topologyParameterWords(options.topology, capacities, options.closedBoundaries,
        options.rigidPressureReaction?.rigidWorldOrigin));
    this.scratch = device.createBuffer({ label: "Losasso compact face publication scratch",
      // Four publication counters plus six packed +/- axis face plans per row.
      // The plans are produced by countLosassoFaces and consumed unchanged by
      // emitLosassoFaces, avoiding a second owner-page traversal.
      size: Math.max(16, capacities.rows * 10 * 4), usage: GPUBufferUsage.STORAGE });
    this.acceptedRigidBoundaryControl = device.createBuffer({
      label: "Losasso accepted rigid-boundary refresh diagnostics",
      size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.acceptedRigidBoundaryControl, 0,
      Uint32Array.of(0, 0, 0, 1, 0, 0));
    this.commitParams = device.createBuffer({ label: "Losasso authority commit parameters",
      // WGSL aligns the trailing vec3u to 16 bytes, so the five scalar words
      // plus padding occupy a 48-byte uniform binding.
      size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.commitParams, 0, Uint32Array.of(capacities.rows,
      capacities.faces, capacities.incidences, capacities.faceAdjacencies,
      this.authority.writable.faceDirectoryCapacity, 0, 0, 0));
    this.allocatedBytes = this.authority.allocatedBytes + this.params.size
      + this.commitParams.size + this.scratch.size + this.acceptedRigidBoundaryControl.size
      + (this.hierarchyPublisher?.allocatedBytes ?? 0) + this.velocityMigration.allocatedBytes;
  }

  get initializationTasks(): readonly LosassoInitializationTask[] {
    return [{ label: "Compile Losasso compact topology publisher", run: () => this.initialize() },
      ...(this.hierarchyPublisher?.initializationTasks ?? []),
      ...this.velocityMigration.initializationTasks];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipelines) return;
    const shaderModule = this.device.createShaderModule({
      label: "Losasso compact topology publication shader",
      code: webgpuOctreeLosassoBackendWGSL,
    });
    this.pipelines = await Promise.all(PUBLISH_ENTRY_POINTS.map((entryPoint) =>
      this.device.createComputePipelineAsync({
        label: `Losasso publication - ${entryPoint}`,
        // Each entry point reaches at most six storage buffers. A shared
        // superset layout would expose thirteen and exceed WebGPU's portable
        // per-stage limit even though no kernel uses that many.
        layout: "auto",
        compute: { module: shaderModule, entryPoint },
      })));
    const commitModule = this.device.createShaderModule({
      label: "Losasso candidate-to-accepted authority commit shader",
      code: webgpuOctreeLosassoAuthorityCommitWGSL,
    });
    this.commitPipelines = await Promise.all(AUTHORITY_COMMIT_ENTRY_POINTS.map((entryPoint) =>
      this.device.createComputePipelineAsync({
        label: `Losasso authority commit - ${entryPoint}`,
        layout: "auto", compute: { module: commitModule, entryPoint },
      })));
  }

  private cachedBindGroup(pipeline: GPUComputePipeline, label: string,
    bindings: readonly number[], buffers: readonly GPUBuffer[]): GPUBindGroup {
    const variants = this.bindGroupCache.get(pipeline) ?? [];
    const cached = variants.find((variant) => variant.bindings.length === bindings.length
      && variant.bindings.every((binding, index) => binding === bindings[index]
        && variant.buffers[index] === buffers[index]));
    if (cached) return cached.group;
    const stableBindings = [...bindings], stableBuffers = [...buffers];
    const group = this.device.createBindGroup({ label,
      layout: pipeline.getBindGroupLayout(0),
      entries: stableBindings.map((binding, index) =>
        ({ binding, resource: { buffer: stableBuffers[index]! } })),
    });
    variants.push({ bindings: stableBindings, buffers: stableBuffers, group });
    this.bindGroupCache.set(pipeline, variants);
    return group;
  }

  encodeCandidatePublication(broker: PassBroker, input: WebGPUOctreeLosassoCandidateInput): void {
    this.assertLive();
    this.encode(broker, input, this.authority.candidate);
  }

  /** Refresh the accepted cut-face aperture and analytic normal velocity
   * without waiting for the next topology transaction. */
  encodeAcceptedRigidBoundaryRefresh(
    broker: PassBroker,
    solidCells: GPUBuffer,
    rigidBodies: GPUBuffer,
  ): void {
    this.assertLive();
    if (!this.pipelines) throw new Error("Losasso topology publisher is not initialized");
    const accepted = this.authority.writable;
    // conditionLosassoFaces only consumes the accepted face count and error
    // word. Snapshot those control words into private storage so refresh-time
    // validation can never mutate or reject the live accepted epoch.
    broker.copyBufferToBuffer(accepted.control, 0,
      this.acceptedRigidBoundaryControl, 0, 24);
    const pipeline = this.pipelines[PUBLISH_ENTRY_POINTS.indexOf("conditionLosassoFaces")]!;
    const bindings = [0, 4, 7, 14, 18, 19] as const;
    const buffers = [this.params, this.acceptedRigidBoundaryControl, accepted.faces,
      accepted.faceGeometry, solidCells, rigidBodies];
    const group = this.acceptedRigidBoundaryGroup ??= this.device.createBindGroup({
      label: "Losasso accepted rigid boundary refresh",
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((binding, index) =>
        ({ binding, resource: { buffer: buffers[index]! } })),
    });
    const pass = broker.compute({ label: "Losasso boundary - refresh accepted rigid faces" });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(accepted.faceDispatch, 0);
  }

  /** Atomically expose a valid candidate through the fixed accepted buffers. */
  encodeReadyCommit(broker: PassBroker, input: {
    readonly frontier: GPUBuffer;
    readonly ownerCandidateTransaction: GPUBuffer;
  }): void {
    this.assertLive();
    if (!this.commitPipelines) throw new Error("Losasso authority commit is not initialized");
    const candidate = this.authority.candidate, accepted = this.authority.writable;
    const buffers = [this.commitParams, candidate.control, input.ownerCandidateTransaction,
      input.frontier, candidate.rowFaceOffsets, accepted.rowFaceOffsets,
      candidate.rowFaces, accepted.rowFaces, candidate.faces, accepted.faces,
      candidate.rowDispatch, accepted.rowDispatch, candidate.faceDispatch,
      accepted.faceDispatch, candidate.rightHandSide, accepted.rightHandSide,
      candidate.faceMetrics, accepted.faceMetrics, candidate.faceAdjacencyOffsets,
      accepted.faceAdjacencyOffsets, candidate.faceAdjacency, accepted.faceAdjacency,
      candidate.extendedVelocity, accepted.extendedVelocity, candidate.faceGeometry,
      accepted.faceGeometry, candidate.axisFaceDirectory, accepted.axisFaceDirectory,
      candidate.diagonal, accepted.diagonal, candidate.solverAuthority,
      accepted.solverAuthority, accepted.control];
    const bindings = [
      [0, 1, 2, 3, 4, 5, 14, 15, 28, 29, 32],
      [0, 1, 2, 3, 4, 6, 7, 32],
      [0, 1, 2, 3, 8, 9, 24, 25, 32],
      [0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 32],
      [0, 1, 2, 3, 22, 23, 32],
      [0, 1, 2, 3, 26, 27, 32],
      [0, 1, 2, 3, 10, 11, 12, 13, 30, 31, 32],
    ] as const;
    const groups = this.commitPipelines.map((pipeline, pipelineIndex) => {
      const selected = bindings[pipelineIndex]!;
      return this.cachedBindGroup(pipeline,
        `Losasso ready authority bindings - ${AUTHORITY_COMMIT_ENTRY_POINTS[pipelineIndex]}`,
        selected, selected.map((binding) => buffers[binding]!));
    });
    const capacities = accepted.capacities;
    const dispatches = [Math.ceil((capacities.rows + 1) / 64),
      Math.ceil(capacities.incidences / 64), Math.ceil(capacities.faces / 64),
      Math.ceil(Math.max(capacities.faces + 1, capacities.faceAdjacencies) / 64),
      Math.ceil(capacities.faces / 64), Math.ceil(accepted.faceDirectoryCapacity / 64), 1];
    const pass = broker.compute({ label: "Losasso topology - commit ready authority bank" });
    const dispatchLimit = this.device.limits.maxComputeWorkgroupsPerDimension;
    for (let index = 0; index < this.commitPipelines.length; index += 1) {
      pass.setPipeline(this.commitPipelines[index]!);
      pass.setBindGroup(0, groups[index]!);
      if (index === 5) pass.dispatchWorkgroupsIndirect(candidate.faceDispatch, 12);
      else {
        const count = dispatches[index]!;
        const width = Math.min(count, dispatchLimit);
        pass.dispatchWorkgroups(width, Math.ceil(count / Math.max(1, width)), 1);
      }
    }
  }

  /** Rebuild owned coarse levels from the accepted, possibly phi-conditioned L0 operator. */
  encodeHierarchyRefresh(broker: PassBroker, acceptedLeafHeaders: GPUBuffer): boolean {
    this.assertLive();
    if (!this.hierarchyPublisher) return false;
    this.hierarchyPublisher.encodeCandidatePublication(broker, {
      leafHeaders: acceptedLeafHeaders,
      finestControl: this.authority.writable.control,
      finestFaces: this.authority.writable.faces,
    });
    return true;
  }

  /** Refresh only accepted coarse face fields; row/face topology stays fixed. */
  encodeHierarchyCoefficientRefresh(broker: PassBroker): boolean {
    this.assertLive();
    if (!this.hierarchyPublisher) return false;
    this.hierarchyPublisher.encodeCoefficientRefresh(broker, {
      control: this.authority.writable.control,
      faces: this.authority.writable.faces,
    });
    return true;
  }

  private encode(broker: PassBroker, input: WebGPUOctreeLosassoCandidateInput,
    output: WebGPUOctreeLosassoWritableAuthority): void {
    this.assertLive();
    if (!this.pipelines) throw new Error("Losasso topology publisher is not initialized");
    if (input.leafHeaders.size < output.capacities.rows * 48) {
      throw new RangeError("Losasso LeafHeader source is smaller than row capacity");
    }
    this.velocityMigration.encodeSnapshot(broker, {
      control: this.authority.writable.control,
      faceDispatch: this.authority.writable.faceDispatch,
      faces: this.authority.writable.faces,
      faceGeometry: this.authority.writable.faceGeometry,
      axisFaceDirectory: this.authority.writable.axisFaceDirectory,
      extendedVelocity: this.authority.writable.extendedVelocity,
    });
    const buffers = [this.params, input.leafHeaders, input.frontier, input.ownerArena,
      output.control, output.rowFaceOffsets, output.rowFaces, output.faces,
      output.rowDispatch, output.faceDispatch, output.rightHandSide,
      output.faceMetrics, output.faceAdjacencyOffsets, this.scratch,
      output.faceGeometry, output.axisFaceDirectory, output.diagonal,
      output.solverAuthority, input.solidCells, input.rigidBodies,
      // Binding 20 is intentionally unused: candidate rows do not have an
      // accepted-index phi. Preserve binding 21's owner-transaction ABI.
      input.ownerCandidateTransaction, input.ownerCandidateTransaction,
      this.authority.writable.control];
    const bindings = [
      [0, 2, 3, 4, 8, 9, 17, 21, 22],
      [0, 1, 3, 4, 10, 13],
      [0, 4, 9, 13],
      [4, 15],
      [0, 1, 3, 4, 7, 11, 13, 14],
      [0, 4, 7, 14, 18, 19],
      [0, 4, 5, 13],
      [0, 4, 5, 6, 7, 13],
      [0, 4, 5, 6, 7, 10, 16],
      [4, 12],
      [4, 14, 15],
      [0, 4, 8, 9, 17],
    ] as const;
    const groups = this.pipelines.map((pipeline, pipelineIndex) => {
      const selected = bindings[pipelineIndex]!;
      return this.cachedBindGroup(pipeline,
        `Losasso compact publication bindings - ${PUBLISH_ENTRY_POINTS[pipelineIndex]}`,
        selected, selected.map((binding) => buffers[binding]!));
    });
    const dispatches = ["single", "row", "single", "directory", "row", "face",
      "single", "face", "row", "face", "face", "single"] as const;
    for (let index = 0; index < this.pipelines.length; index += 1) {
      // The eight-byte solid-cell sentinel means this scene has neither
      // terrain nor rigid geometry. emitLosassoFaces already publishes the
      // exact all-open (or closed world-wall) coefficients for that case, so
      // the span-by-span analytic aperture fold would only recompute 1/0 and
      // zero normal velocity for every face.
      if (index === 5 && input.solidCells.size <= 8) continue;
      // These labels collapse into the already-open production pass. Under
      // explicit label-isolation profiling they expose the individual compact
      // publication stages without changing shipping encoder shape.
      const pass = broker.compute({ label:
        `Losasso topology - ${PUBLISH_ENTRY_POINTS[index]}` });
      pass.setPipeline(this.pipelines[index]!);
      pass.setBindGroup(0, groups[index]!);
      const dispatch = dispatches[index]!;
      if (dispatch === "row") pass.dispatchWorkgroupsIndirect(output.rowDispatch, 0);
      else if (dispatch === "face") pass.dispatchWorkgroupsIndirect(output.faceDispatch, 0);
      else if (dispatch === "directory") {
        pass.dispatchWorkgroupsIndirect(output.faceDispatch, 12);
      }
      else pass.dispatchWorkgroups(1, 1, 1);
    }
    this.velocityMigration.encodeMigration(broker, {
      control: output.control,
      faceDispatch: output.faceDispatch,
      faces: output.faces,
      faceGeometry: output.faceGeometry,
      axisFaceDirectory: output.axisFaceDirectory,
      extendedVelocity: output.extendedVelocity,
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
    this.commitParams.destroy();
    this.scratch.destroy();
    this.hierarchyPublisher?.destroy();
    this.velocityMigration.destroy();
    this.authority.destroy();
    this.acceptedRigidBoundaryControl.destroy();
    this.bindGroupCache.clear();
    this.acceptedRigidBoundaryGroup = undefined;
    this.pipelines = undefined;
    this.commitPipelines = undefined;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso topology publisher is destroyed");
  }
}

/** Concrete wide exact-reduction solver adapter over the reduced diagonal ABI. */
class WebGPUOctreeLosassoWideSolver implements WebGPUOctreeLosassoPressureSolver {
  readonly initializationTasks: readonly LosassoInitializationTask[];
  readonly allocatedBytes: number;
  readonly vectors: OctreePipelinedMGPCGVectors;
  private readonly executor: WebGPUOctreePipelinedMGPCG;
  get control(): GPUBuffer { return this.executor.control; }
  get iterationBudget(): number { return this.executor.iterationBudget; }
  get symmetryStageAuditBuffers() { return this.executor.symmetryStageAuditBuffers; }

  constructor(device: GPUDevice, input: {
    readonly rowCapacity: number;
    readonly diagonal: GPUBuffer;
    readonly rightHandSide: GPUBuffer;
    readonly control: GPUBuffer;
    readonly acceptedAuthority: GPUBuffer;
    readonly rowDispatch: GPUBuffer;
    readonly operator: WebGPUOctreeLosassoOperator;
    readonly preconditioner: OctreeLosassoFirstOrderVCycle;
    readonly options?: Omit<OctreePipelinedMGPCGOptions, "rowCapacity">;
  }) {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      | GPUBufferUsage.COPY_SRC;
    const vector = (label: string) => device.createBuffer({
      label: `Losasso wide MGPCG ${label}`, size: input.rowCapacity * 4, usage: storage,
    });
    this.vectors = Object.freeze({
      pressure: vector("pressure"), residual: vector("residual"),
      preconditioned: vector("preconditioned residual"),
      preconditionedImage: vector("preconditioned image"),
      direction: vector("direction"), directionImage: vector("direction image"),
    });
    this.executor = new WebGPUOctreePipelinedMGPCG(device, {
      diagonal: input.diagonal,
      diagonalStrideWords: 1,
      diagonalBankStrideWords: 0,
      rhs: input.rightHandSide,
      rowCount: input.control,
      rowDispatch: input.rowDispatch,
      acceptedAuthority: input.acceptedAuthority,
      operator: input.operator,
      preconditioner: input.preconditioner,
      vectors: this.vectors,
    }, { rowCapacity: input.rowCapacity, ...input.options });
    this.initializationTasks = [{ label: "Compile Losasso wide exact-reduction MGPCG",
      run: () => this.executor.initializePipelines() }];
    this.allocatedBytes = this.executor.allocatedBytes
      + Object.values(this.vectors).reduce((sum, buffer) => sum + buffer.size, 0);
  }

  encodeSolve(broker: PassBroker, input: {
    readonly pressureSeed: GPUBuffer;
    readonly pressureOut: GPUBuffer;
    readonly rightHandSide: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void {
    this.executor.encode(broker, { pressureSeed: input.pressureSeed,
      pressureOut: input.pressureOut });
  }

  setSolveTuning(tuning: OctreeLosassoSolveTuning): void {
    this.executor.setSolveTuning(tuning);
  }

  destroy(): void {
    this.executor.destroy();
    for (const buffer of Object.values(this.vectors)) buffer.destroy();
  }
}

/**
 * Constructible pipeline seam for the Losasso coarse side. It owns the
 * reduced face authority and topology publisher, then exposes the operator,
 * V-cycle, pressure solver, axis-face projection, and fixed-K extension as one
 * lifetime. No Power resource type is imported by this module.
 */
export class WebGPUOctreeLosassoCoarseBackend {
  readonly kind = "losasso" as const;
  readonly gradingPolicy = "uniform-fine-free-surface-shell" as const;
  readonly operator: WebGPUOctreeLosassoOperator;
  readonly preconditioner: OctreeLosassoFirstOrderVCycle;
  readonly solver: WebGPUOctreeLosassoPressureSolver;
  readonly dynamics: WebGPUOctreeLosassoDynamics;
  readonly projection: WebGPUOctreeLosassoProjection;
  readonly extensionBand: WebGPUOctreeLosassoExtensionBand;
  readonly rigidPressureReaction?: WebGPUOctreeLosassoRigidPressureReaction;
  readonly sources: WebGPUOctreeLosassoPublishedSources;
  readonly allocatedBytes: number;
  readonly authorityPublication = "staged-fail-closed" as const;
  readonly extensionPublication = "W7-finest-air-face-band" as const;
  get solverControl(): GPUBuffer | undefined { return this.solver.control; }
  get solverIterationBudget(): number | undefined { return this.solver.iterationBudget; }
  get solverSymmetryStageAuditBuffers() { return this.solver.symmetryStageAuditBuffers; }

  /**
   * Adopt live coarse-band accuracy/cost dials.
   *
   * Everything here is a queue write into a buffer this backend already owns —
   * no pipeline is recompiled, no allocation moves, and the accepted topology
   * is untouched — so the caller may do this between advances while the
   * simulation runs. The V-cycle sweep count reaches the preconditioner
   * directly as well as the solver, because the resident executor inlines the
   * cycle while the wide executor dispatches this object's kernels.
   */
  applySolveTuning(tuning: OctreeLosassoSolveTuning): void {
    this.solver.setSolveTuning?.(tuning);
    if (!(this.preconditioner instanceof WebGPUOctreeLosassoVCycle)) return;
    if (tuning.bottomSweeps !== undefined) {
      this.preconditioner.setBottomSweeps(tuning.bottomSweeps);
    }
    if (tuning.smoothingSweeps !== undefined) {
      this.preconditioner.setSmoothingSweeps(tuning.smoothingSweeps);
    }
  }

  /** Section 5 axis-face extension sweeps per advance. */
  setVelocityExtensionSweeps(sweeps: number): void {
    this.extensionBand.setVelocityExtensionSweeps(sweeps);
  }

  private readonly publisher: WebGPUOctreeLosassoTopologyPublisher;
  private initialized = false;
  private destroyed = false;

  constructor(private readonly options: WebGPUOctreeLosassoBackendOptions) {
    this.publisher = new WebGPUOctreeLosassoTopologyPublisher(options.device, options);
    const published = this.publisher.sources;
    this.extensionBand = new WebGPUOctreeLosassoExtensionBand(options.device, {
      dimensions: options.topology.dimensions,
      maximumLeafSize: options.topology.maximumLeafSize,
      cellSize: options.topology.physicalCellSize[0],
      domainOrigin: options.topology.domainOrigin,
      wetFaceCapacity: options.capacities.faces,
      maximumResidentFineBricks: options.extensionBandBrickCapacity,
      velocityExtensionMode: options.velocityExtensionMode,
      closedBoundaries: options.closedBoundaries,
      wet: {
        control: published.operator.control,
        faceDispatch: published.dynamics.faceDispatch,
        faceGeometry: published.dynamics.faceGeometry,
        axisFaceDirectory: published.dynamics.axisFaceDirectory,
        directoryCapacity: published.dynamics.faceDirectoryCapacity,
        projectedVelocity: published.projection.projectedVelocity,
        extendedVelocity: published.extension.extendedVelocity,
      },
    });
    this.sources = Object.freeze({ ...published,
      velocitySampler: this.extensionBand.samplerSource });
    this.operator = new WebGPUOctreeLosassoOperator(options.device, this.sources.operator);
    const cycle = options.createVCycle
      ? options.createVCycle(this.sources.vcycle)
      : new WebGPUOctreeLosassoVCycle(options.device, this.sources.vcycle);
    assertOctreeLosassoVCycle(cycle);
    this.preconditioner = cycle;
    this.solver = options.createSolver
      ? options.createSolver({ operator: this.operator, preconditioner: cycle })
      : options.residentSolver && WebGPUOctreeLosassoResidentMGPCG.supports(
        options.device, this.sources.vcycle, options.capacities.rows)
        ? new WebGPUOctreeLosassoResidentMGPCG(options.device, {
          rowCapacity: options.capacities.rows,
          diagonal: this.publisher.authority.writable.diagonal,
          rightHandSide: this.sources.rightHandSide,
          acceptedAuthority: this.publisher.authority.writable.solverAuthority,
          hierarchy: this.sources.vcycle,
        }, {
          maximumIterations: options.solver?.hardIterationCeiling
            ?? options.solver?.maximumIterations ?? 40,
          ...(options.solver?.relativeTolerance === undefined
            ? {} : { relativeTolerance: options.solver.relativeTolerance }),
          ...(options.solver?.absoluteTolerance === undefined
            ? {} : { absoluteTolerance: options.solver.absoluteTolerance }),
        })
        : new WebGPUOctreeLosassoWideSolver(options.device, {
          rowCapacity: options.capacities.rows,
          diagonal: this.publisher.authority.writable.diagonal,
          rightHandSide: this.sources.rightHandSide,
          control: this.publisher.authority.writable.control,
          acceptedAuthority: this.publisher.authority.writable.solverAuthority,
          rowDispatch: this.publisher.authority.writable.rowDispatch,
          operator: this.operator,
          preconditioner: cycle,
          options: options.solver,
        });
    this.dynamics = new WebGPUOctreeLosassoDynamics(options.device,
      this.sources.dynamics, {
        dimensions: options.topology.dimensions,
        maximumLeafSize: options.topology.maximumLeafSize,
        physicalCellSize: options.topology.physicalCellSize,
        domainOrigin: options.topology.domainOrigin,
        density: options.density,
        closedBoundaries: options.closedBoundaries,
      }, {
        control: this.extensionBand.source.control,
        faceGeometry: this.extensionBand.samplerSource.faceGeometry,
        axisFaceDirectory: this.extensionBand.samplerSource.axisFaceDirectory,
        extendedVelocity: this.extensionBand.source.extendedVelocity,
        predictorExtendedVelocity: this.extensionBand.predictorVelocity,
        stagedVelocity: this.extensionBand.dynamicsStagedVelocity,
        predictorStagedVelocity: this.extensionBand.predictorStagedVelocity,
        faceCapacity: this.extensionBand.plan.faceCapacity,
        directoryCapacity: this.extensionBand.plan.directoryCapacity,
      });
    this.projection = new WebGPUOctreeLosassoProjection(options.device,
      this.sources.projection, {
        density: options.density,
        physicalCellSize: options.topology.physicalCellSize[0],
        freeSurfacePressureMode: options.freeSurfacePressureMode,
      });
    if (options.rigidPressureReaction) {
      this.rigidPressureReaction = new WebGPUOctreeLosassoRigidPressureReaction(options.device, {
        rowCapacity: options.capacities.rows,
        faceCapacity: options.capacities.faces,
        faceDispatch: this.publisher.authority.writable.faceDispatch,
        faces: published.operator.faces,
        faceGeometry: published.dynamics.faceGeometry,
        solidCells: options.rigidPressureReaction.solidCells,
        rigidBodies: options.rigidPressureReaction.rigidBodies,
        rigidExchange: options.rigidPressureReaction.rigidExchange,
      }, {
        dimensions: options.topology.dimensions,
        physicalCellSize: options.topology.physicalCellSize,
        rigidWorldOrigin: options.rigidPressureReaction.rigidWorldOrigin,
        density: options.density,
        hydrostaticReferenceY_m: options.rigidPressureReaction.hydrostaticReferenceY_m,
      });
    }
    if (this.sources.projection.projectedVelocity
      !== this.sources.extension.projectedVelocity) {
      throw new Error("Losasso projection output must be the fixed-K extension seed field");
    }
    const preconditionerBytes = (cycle as OctreeLosassoFirstOrderVCycle &
      { allocatedBytes?: number }).allocatedBytes ?? 0;
    this.allocatedBytes = this.publisher.allocatedBytes + this.dynamics.allocatedBytes
      + this.extensionBand.allocatedBytes
      + (this.rigidPressureReaction?.allocatedBytes ?? 0)
      + 16 + (this.solver.allocatedBytes ?? 0) + preconditionerBytes;
  }

  get initializationTasks(): readonly LosassoInitializationTask[] {
    return [
      ...this.publisher.initializationTasks,
      ...this.operator.initializationTasks,
      ...(this.preconditioner as OctreeLosassoFirstOrderVCycle & {
        initializationTasks?: readonly LosassoInitializationTask[];
      }).initializationTasks ?? [],
      ...(this.solver.initializationTasks ?? []),
      ...this.dynamics.initializationTasks,
      ...this.projection.initializationTasks,
      ...(this.rigidPressureReaction?.initializationTasks ?? []),
      ...this.extensionBand.initializationTasks,
    ];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.initialized) return;
    for (const task of this.initializationTasks) await task.run();
    this.initialized = true;
  }

  /** Candidate construction and ready publication are one fail-closed GPU pass. */
  encodeCandidatePublication(broker: PassBroker,
    input: WebGPUOctreeLosassoCandidateInput): void {
    this.assertReady();
    this.publisher.encodeCandidatePublication(broker, input);
  }

  /** Candidate authority used only by the ready validator. */
  get candidateAuthorityControl(): GPUBuffer { return this.publisher.authority.candidate.control; }
  /** Refresh-only validation control; never aliases accepted authority. */
  get rigidBoundaryRefreshDiagnostics(): GPUBuffer {
    return this.publisher.acceptedRigidBoundaryDiagnostics;
  }

  encodeReadyCommit(broker: PassBroker, input: {
    readonly frontier: GPUBuffer;
    readonly ownerCandidateTransaction: GPUBuffer;
  }): void {
    this.assertReady();
    this.publisher.encodeReadyCommit(broker, input);
    // The W7 graph spans topology epochs, but its cached wet ids do not. Refresh
    // them through the dense finest-face owner map at the topology boundary so
    // both projected and MacCormack gathers stay direct in the advance path.
    this.extensionBand.encodeTopologyRemap(broker);
  }

  /** Call immediately after accepted ghost conditioning changes L0 coefficients. */
  encodeHierarchyRefresh(broker: PassBroker, acceptedLeafHeaders: GPUBuffer): boolean {
    this.assertReady();
    return this.publisher.encodeHierarchyRefresh(broker, acceptedLeafHeaders);
  }


  /** Call when ghost conditioning changed fields without changing topology. */
  encodeHierarchyCoefficientRefresh(broker: PassBroker): boolean {
    this.assertReady();
    return this.publisher.encodeHierarchyCoefficientRefresh(broker);
  }

  encodeSolve(broker: PassBroker, input: {
    readonly pressureSeed: GPUBuffer;
    readonly pressureOut: GPUBuffer;
  }): void {
    this.assertReady();
    this.solver.encodeSolve(broker, { ...input, rightHandSide: this.sources.rightHandSide,
      rowCount: this.sources.rowCount });
  }

  encodeAdvection(broker: PassBroker, step: WebGPUOctreeLosassoDynamicsStep): void {
    this.assertReady();
    this.dynamics.encodeAdvection(broker, step, () =>
      this.extensionBand.encodePredictorExtension(
        broker, this.sources.dynamics.advectedVelocity,
      ));
  }

  encodeRigidBoundaryRefresh(broker: PassBroker): boolean {
    this.assertReady();
    const rigid = this.options.rigidPressureReaction;
    if (!rigid) return false;
    this.publisher.encodeAcceptedRigidBoundaryRefresh(
      broker, rigid.solidCells, rigid.rigidBodies,
    );
    return true;
  }

  encodeForcesAndDivergence(
    broker: PassBroker,
    step: WebGPUOctreeLosassoDynamicsStep,
  ): void {
    this.assertReady();
    this.dynamics.encodeForcesAndDivergence(broker, step);
  }

  encodeProjection(
    broker: PassBroker,
    pressure: GPUBuffer,
    step: WebGPUOctreeLosassoDynamicsStep,
    dynamicCouplingBodyCount = 0,
  ): void {
    this.assertReady();
    this.projection.encode(broker, pressure, step.dt_s / this.options.density,
      step.gravity_m_s2);
    this.rigidPressureReaction?.encode(
      broker, pressure, step.dt_s, dynamicCouplingBodyCount, step.gravity_m_s2,
    );
    this.dynamics.encodeInflowConstraint(broker, step);
  }

  /** Publish the factor-4/8 phi-classified W=7 graph before its first S3e use. */
  encodeExtensionBandPublication(broker: PassBroker,
    fine: WebGPUFineLevelSetBrickSource): void {
    this.assertReady();
    this.extensionBand.encodePublication(broker, fine);
  }

  encodeCoarseExtensionBandPublication(
    broker: PassBroker,
    coarsePhi: WebGPUOctreeLosassoCoarsePhiSource,
    generation: number,
  ): void {
    this.assertReady();
    this.extensionBand.encodeCoarsePublication(broker, coarsePhi, generation);
  }

  encodeExtension(broker: PassBroker, advance: number, topologyEpoch: number): boolean {
    this.assertReady();
    return this.extensionBand.encodeOncePerAdvance(broker, advance, topologyEpoch);
  }

  get extensionBandPublished(): boolean { return this.extensionBand.hasPublishedGraph; }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.solver.destroy();
    const preconditioner = this.preconditioner as OctreeLosassoFirstOrderVCycle
      & { destroy?: () => void };
    preconditioner.destroy?.();
    this.operator.destroy();
    this.dynamics.destroy();
    this.projection.destroy();
    this.rigidPressureReaction?.destroy();
    this.extensionBand.destroy();
    this.publisher.destroy();
  }

  private assertReady(): void {
    this.assertLive();
    if (!this.initialized) throw new Error("Losasso coarse backend is not initialized");
  }
  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso coarse backend is destroyed");
  }
}
