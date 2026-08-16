import { GPUInitializationTaskRunner } from "../../core/gpu-initialization";
import type { GPUQuality } from "../../core/gpu-quality";
import type {
  GPUInitializationReporter,
  GPUSolverInstance,
  InjectedLiquidBall,
  MethodParamValues,
} from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import { classifyFineBoxAgainstSphericalContainer } from "../../core/spherical-container";
import { initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import { GPUStageTimestampRecorder } from "../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  type GPUEulerianInfo,
  type GPURigidLoad,
} from "../../core/webgpu-eulerian";
import { WebGPURigidBodySystem } from "../../core/webgpu-rigid-body";
import { sceneHasTerrain, terrainColumnHeights } from "../../core/terrain";
import {
  ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS,
  AdaptiveMassFrameCapture,
} from "./adaptive-mass-frame-pipeline";
import type { AdaptiveMassSolverOptions } from "./method";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  sparseBrickContainingCoordinate,
  sparseBrickKey,
  sparseBrickSpan,
  sparseBrickAtlasStats,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "./sparse-brick-atlas";
import {
  buildSparseAtlasCompositeGrid,
  type SparseAtlasCompositeGrid,
} from "./sparse-atlas-composite-projection";
import { WebGPUAdaptiveMassSparsePresentation } from
  "./webgpu-adaptive-mass-atlas-presentation";
import {
  sparseCM12ActivityPolicy,
  sparseCM12PressureIterations,
  sparseCM12PressureRelativeTolerance,
  sparseCM12SharpeningDistance,
  sparseCM12SharpeningTraceSteps,
  WebGPUSparseCM12Resident,
  type SparseCM12GPUActivityRecord,
} from "./webgpu-sparse-cm12-resident";

/** Method-local long-run physics receipt carried through the generic info bag. */
export interface AdaptiveMassStepTelemetry {
  adaptiveKineticEnergyBeforeFineUnits?: number;
  adaptiveKineticEnergyAfterFineUnits?: number;
  adaptiveProjectionKineticEnergyBeforeFineUnits?: number;
  adaptiveProjectionKineticEnergyAfterFineUnits?: number;
  adaptiveInactiveFaceCount?: number;
  adaptiveMaximumInactiveFaceSpeedBefore_m_s?: number;
  adaptiveMaximumInactiveFaceSpeedAfter_m_s?: number;
  adaptiveMaximumMixedSeamDivergence_s?: number;
  adaptiveMaximumDensityAfterTransport?: number;
  adaptiveMaximumDensityAfterConditioning?: number;
}

export interface AdaptiveMassGPUActivityBrick extends SparseCM12GPUActivityRecord {
  readonly key: number;
  readonly coordinate: SparseBrickVec3;
  readonly resolution: SparseBrickResolution;
}

/**
 * Read-only receipt from the device scheduler. `advanceTo` never consumes this
 * shape: it exists only so explicit diagnostics can explain what the fixed GPU
 * dispatch chain accepted and what remains queued.
 */
interface SparseCM12TopologySchedulerDiagnostics {
  readonly acceptedTopologyGeneration?: number;
  readonly topologyUrgentQueuedBrickCount?: number;
  readonly topologyOrdinaryQueuedBrickCount?: number;
  readonly topologyPreparedBrickCount?: number;
  readonly topologyCommittedBrickCount?: number;
  readonly topologyDeferredBrickCount?: number;
  readonly acceptedFineBrickCount?: number;
  readonly acceptedCoarseBrickCount?: number;
}

/** Select a fine receiver and its strongly graded outward support rung. */
export function dormantReceiverResolution(
  mode: AdaptiveMassSolverOptions["resolutionMode"],
  distance = 0,
  sourceResolution: SparseBrickResolution = 8,
): SparseBrickResolution {
  if (mode === "all-fine") return 8;
  if (mode === "all-coarse") return 4;
  let resolution = sourceResolution;
  for (let step = 0; step < distance; step += 1) {
    resolution = resolution === 8 ? 4 : resolution === 4 ? 2 : 1;
  }
  return resolution;
}

/**
 * Fixed construction-time capacity reach of the GPU-resident receiver pool.
 *
 * This is a capacity bound, not a domain bound. It reserves a local apron far
 * enough for several brick crossings; the GPU
 * scheduler may split and merge bricks within that capacity after attachment.
 * Crucially, a kilometre-wide authored
 * world costs the same as a small one when their live liquid sets are equal.
 */
export const SPARSE_CM12_RECEIVER_SUPPORT_RINGS = 9;
/** Construction capacity is proportional to represented leaves, never domain volume. */
export const SPARSE_CM12_RECEIVER_CAPACITY_FACTOR = 4;
export const SPARSE_CM12_MINIMUM_RECEIVER_CAPACITY = 512;

/**
 * Keep the construction-time receiver apron fixed in metres when the editor
 * moves a scene along its DETAIL axis. `nominalResolution` follows WORLD edits
 * but deliberately stays put for DETAIL edits, so their ratio is the exact
 * linear lattice scale and the receiver volume scales by its cube.
 */
export function adaptiveMassReceiverScaleForScene(
  scene: Pick<SceneDescription, "nominalResolution" | "voxelDomain">,
  supportRings = SPARSE_CM12_RECEIVER_SUPPORT_RINGS,
): { readonly supportRings: number; readonly minimumCapacityScale: number } {
  if (!Number.isSafeInteger(supportRings) || supportRings < 0) {
    throw new RangeError("Sparse CM12 receiver support rings must be a non-negative integer");
  }
  const detailScale = scene.nominalResolution.length_m
    / scene.voxelDomain.finestCellSize_m;
  const finiteScale = Number.isFinite(detailScale) && detailScale > 0 ? detailScale : 1;
  return {
    // residentSupportAtlas contributes one immediate face-receiver shell
    // before this dormant apron is grown. Scale their combined physical reach,
    // then remove that still-one-brick active shell.
    supportRings: Math.max(0, Math.round((supportRings + 1) * finiteScale) - 1),
    minimumCapacityScale: finiteScale ** 3,
  };
}

function receiverBoundaryResolution(
  atlas: SparseAdaptiveMassAtlas,
  coordinate: SparseBrickVec3,
): SparseBrickResolution | undefined {
  if (!atlas.boundary) return undefined;
  const minimum = coordinate.map((value) => value * 8) as [number, number, number];
  const maximum = coordinate.map((value) => (value + 1) * 8) as [number, number, number];
  const classification = classifyFineBoxAgainstSphericalContainer(
    atlas.boundary, minimum, maximum,
  );
  if (classification === "outside") return undefined;
  if (classification === "cut") return 4;
  const collarMinimum = minimum.map((value) => value - 8) as [number, number, number];
  const collarMaximum = maximum.map((value) => value + 8) as [number, number, number];
  return classifyFineBoxAgainstSphericalContainer(
    atlas.boundary, collarMinimum, collarMaximum,
  ) === "cut" ? 2 : 1;
}

function closeReceiverGrading(
  source: SparseAdaptiveMassAtlas,
  bricks: Map<number, SparseAdaptiveMassBrick>,
): void {
  let changed = true;
  while (changed) {
    changed = false;
    for (const brick of [...bricks.values()]) for (let axis = 0; axis < 3; axis += 1) {
      if (sparseBrickSpan(brick) > 1) continue;
      const coordinate = [...brick.coordinate] as [number, number, number];
      coordinate[axis] += 1;
      if (coordinate[axis] >= source.brickDimensions[axis]) continue;
      const neighbor = bricks.get(sparseBrickKey(coordinate, source.brickDimensions));
      if (!neighbor || sparseBrickSpan(neighbor) > 1) continue;
      const high = Math.max(brick.resolution, neighbor.resolution);
      const low = brick.resolution < neighbor.resolution ? brick : neighbor;
      if (high <= 2 * low.resolution) continue;
      const resolution = (high / 2) as SparseBrickResolution;
      const density = new Float64Array(resolution ** 3);
      const gamma = new Float64Array(resolution ** 3);
      for (let z = 0; z < resolution; z += 1)
        for (let y = 0; y < resolution; y += 1)
          for (let x = 0; x < resolution; x += 1) {
            const local = x + resolution * (y + resolution * z);
            const sx = Math.min(low.resolution - 1, Math.floor(x * low.resolution / resolution));
            const sy = Math.min(low.resolution - 1, Math.floor(y * low.resolution / resolution));
            const sz = Math.min(low.resolution - 1, Math.floor(z * low.resolution / resolution));
            const sourceLocal = sx + low.resolution * (sy + low.resolution * sz);
            density[local] = low.density[sourceLocal]!;
            gamma[local] = low.gamma[sourceLocal]!;
          }
      bricks.set(low.key, { ...low, resolution, density, gamma });
      changed = true;
    }
  }
}

/** Reserve a compact receiver halo for the fixed-topology GPU control arms. */
function brickFaceCarriesFluid(
  brick: SparseAdaptiveMassBrick,
  direction: SparseBrickVec3,
): boolean {
  const resolution = brick.resolution;
  const axis = direction[0] !== 0 ? 0 : direction[1] !== 0 ? 1 : 2;
  const fixed = direction[axis] < 0 ? 0 : resolution - 1;
  for (let second = 0; second < resolution; second += 1)
    for (let first = 0; first < resolution; first += 1) {
      const coordinate = axis === 0 ? [fixed, first, second]
        : axis === 1 ? [first, fixed, second] : [first, second, fixed];
      const at = coordinate[0] + resolution
        * (coordinate[1] + resolution * coordinate[2]);
      if (brick.density[at]! > 0) return true;
    }
  return false;
}

export function residentSupportAtlas(
  source: SparseAdaptiveMassAtlas,
  mode: AdaptiveMassSolverOptions["resolutionMode"],
): SparseAdaptiveMassAtlas {
  const bricks = new Map(source.bricks.map((brick) => [brick.key, brick] as const));
  // Construction starts the immediate shell fine so the first transport step
  // has the same destination fidelity as the represented interface. Activity
  // can subsequently coarsen static-only support; characteristic-swept empty
  // destinations retain the 8^3 floor.
  const receiverResolution: SparseBrickResolution = mode === "all-coarse" ? 4 : 8;
  for (const brick of source.bricks) {
    if (sparseBrickSpan(brick) > 1) continue;
    for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1)
      for (let dx = -1; dx <= 1; dx += 1) {
        // Conservative transport crosses a face. Reserve the six immediate
        // face receivers at the mandatory 8^3 floor; edge/corner support is
        // supplied by the graded dormant apron below. Promoting the whole
        // 3x3x3 cube made a curved Figure 7 source turn 400 bricks fine before
        // its first step and dominated scene construction.
        if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) !== 1) continue;
        const coordinate: SparseBrickVec3 = [
          brick.coordinate[0] + dx,
          brick.coordinate[1] + dy,
          brick.coordinate[2] + dz,
        ];
        if (coordinate.some((value, axis) => value < 0
          || value >= source.brickDimensions[axis])) continue;
        const key = sparseBrickKey(coordinate, source.brickDimensions);
        if (bricks.has(key) || sparseBrickContainingCoordinate(source, coordinate)) continue;
        if (!brickFaceCarriesFluid(brick, [dx, dy, dz])) continue;
        const boundaryResolution = receiverBoundaryResolution(source, coordinate);
        if (source.boundary && boundaryResolution === undefined) continue;
        const resolution = Math.max(receiverResolution,
          boundaryResolution ?? 1) as SparseBrickResolution;
        const count = resolution ** 3;
        const receiver: SparseAdaptiveMassBrick = {
          key, coordinate, resolution,
          density: new Float64Array(count),
          gamma: new Float64Array(count).fill(1),
        };
        bricks.set(key, receiver);
      }
  }
  closeReceiverGrading(source, bricks);
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key),
    source.generation,
    source.boundary,
  );
}

/**
 * Preallocate dormant receivers over a bounded apron around retained bricks.
 * A one-axis corridor leaves corner-authored dams capped by the transverse
 * support halo (cell 47 in the canonical 64-cubed mini dam).
 */
export function dormantReceiverDomain(
  source: SparseAdaptiveMassAtlas,
  mode: AdaptiveMassSolverOptions["resolutionMode"],
  supportRings = SPARSE_CM12_RECEIVER_SUPPORT_RINGS,
  receiverFloor: "auto" | SparseBrickResolution = "auto",
  minimumCapacityScale = 1,
): SparseAdaptiveMassAtlas {
  if (!Number.isSafeInteger(supportRings) || supportRings < 0) {
    throw new RangeError("Sparse CM12 receiver support rings must be a non-negative integer");
  }
  if (!Number.isFinite(minimumCapacityScale) || minimumCapacityScale <= 0) {
    throw new RangeError("Sparse CM12 receiver capacity scale must be positive and finite");
  }
  const bricks = new Map(source.bricks.map((brick) => [brick.key, brick] as const));
  const maximumReceiverBricks = Math.min(
    source.brickDimensions.reduce((product, value) => product * value, 1),
    Math.max(Math.ceil(SPARSE_CM12_MINIMUM_RECEIVER_CAPACITY * minimumCapacityScale),
      SPARSE_CM12_RECEIVER_CAPACITY_FACTOR * source.bricks.length),
  );
  // Multi-source breadth-first growth visits exactly the retained apron.  The
  // previous three nested domain loops made an empty 128^3 Figure 7 tank build
  // 4,096 fine bricks and made construction scale with empty world volume.
  const distanceByKey = new Map<number, number>();
  const queue: SparseBrickVec3[] = [];
  // Boundary-fed liquid (dams, inlets and full-depth slabs) has an authored
  // escape direction and can cross the whole fixed receiver apron. Its cold
  // pool needs 4^3 physical rows to keep the advancing body from collapsing
  // into a few immutable coarse cells. Interior droplets retain the sparse
  // 4/2/1 cold hierarchy; applying 4^3 to their entire 19-brick-wide
  // neighbourhood made Figure 7 construction almost twice as expensive and
  // measurably increased its frame time.
  const boundaryFed = !source.boundary && source.bricks.some((brick) => brick.density.some(
    (density) => density > 0,
  ) && brick.coordinate.some((value, axis) => value === 0
    || value === source.brickDimensions[axis] - 1));
  const physicalReceiverFloor: SparseBrickResolution = mode === "all-fine" ? 8
    : mode === "all-coarse" ? 4
      : receiverFloor === "auto" ? (boundaryFed ? 4 : 1) : receiverFloor;
  // If the complete authored tank already fits inside the leaf budget, retain
  // it. A fixed ring count otherwise creates a hidden numerical wall one brick
  // before the real wall (Figure 9's 40-cell reservoir plus nine rings stopped
  // at x=119 in a 128-cell tank). Large worlds still use the caller's bounded
  // reach because their logical brick volume exceeds this same capacity.
  const logicalBrickCount = source.brickDimensions.reduce(
    (product, value) => product * value, 1,
  );
  const effectiveSupportRings = boundaryFed && logicalBrickCount <= maximumReceiverBricks
    ? Math.max(...source.brickDimensions)
    : supportRings;
  for (const brick of source.bricks) {
    distanceByKey.set(brick.key, 0);
    if (sparseBrickSpan(brick) === 1) queue.push(brick.coordinate);
  }
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const coordinate = queue[cursor]!;
    const ownKey = sparseBrickKey(coordinate, source.brickDimensions);
    const distance = distanceByKey.get(ownKey)!;
    if (distance >= effectiveSupportRings) continue;
    for (let dz = -1; dz <= 1; dz += 1)
      for (let dy = -1; dy <= 1; dy += 1)
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          const neighbor: SparseBrickVec3 = [
            coordinate[0] + dx,
            coordinate[1] + dy,
            coordinate[2] + dz,
          ];
          if (neighbor.some((value, axis) => value < 0
            || value >= source.brickDimensions[axis])) continue;
          const key = sparseBrickKey(neighbor, source.brickDimensions);
          if (distanceByKey.has(key)) continue;
          const containingSource = sparseBrickContainingCoordinate(source, neighbor);
          if (containingSource && sparseBrickSpan(containingSource) > 1) continue;
          if (!bricks.has(key) && !containingSource
            && bricks.size >= maximumReceiverBricks) continue;
          const boundaryResolution = receiverBoundaryResolution(source, neighbor);
          if (source.boundary && boundaryResolution === undefined) continue;
          const nextDistance = distance + 1;
          distanceByKey.set(key, nextDistance);
          queue.push(neighbor);
          if (bricks.has(key) || containingSource) continue;
          // Packed composite rows are immutable while the solver is attached.
          // Until a candidate topology can replace those rows, a 2^3/1^3
          // dormant brick would remain that coarse after the GPU planner asks
          // for 8^3, concentrating an advancing dam front into one giant cell.
          // This is still capacity/apron-shaped rather than world-shaped. The
          // physical floor is selected once from the retained liquid source,
          // never from logical-domain volume or a domain scan.
          const planned = dormantReceiverResolution(mode, nextDistance);
          let resolution: SparseBrickResolution = mode === "adaptive"
            ? Math.max(physicalReceiverFloor, planned) as SparseBrickResolution : planned;
          resolution = Math.max(resolution, boundaryResolution ?? 1) as SparseBrickResolution;
          const count = resolution ** 3;
          bricks.set(key, {
            key, coordinate: neighbor, resolution,
            density: new Float64Array(count),
            gamma: new Float64Array(count).fill(1),
          });
        }
  }
  closeReceiverGrading(source, bricks);
  return createSparseAdaptiveMassAtlas(
    source.dimensions,
    [...bricks.values()].sort((left, right) => left.key - right.key),
    source.generation,
    source.boundary,
  );
}

/**
 * GPU-resident Sparse CM12 authority. Construction may build compact topology
 * on the host, but every accepted frame is device-only simulation work: the
 * host writes one small uniform block and encodes a fixed dispatch schedule.
 */
export class WebGPUAdaptiveMassSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly gridCellTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly gridPressureTexture: GPUTexture;
  readonly gridDivergenceTexture: GPUTexture;
  readonly initialSparseAuthorityReady = true;
  get sparseAdaptiveGridSource() { return this.resident.sparseAdaptiveGridSource; }
  get tracerSource() { return this.resident.tracerSource; }
  setTracersEnabled(enabled: boolean) { this.resident.setTracersEnabled(enabled); }
  /** Re-read the mixing from now; the marker colours re-date to this frame. */
  reseedTracers() { this.resident.reseedTracers(); }
  /** QA receipt: `[x, y, z, live]` per marker, in fine-lattice units. */
  readTracers() { return this.resident.readTracers(); }
  /** Face velocities and the row records that place them; no enable needed. */
  get faceVelocitySource() { return this.resident.faceVelocitySource; }
  /**
   * The pressure lab's capture controls.
   *
   * Present only when the solver was built with `pressureJournal`, because the
   * journal is a construction-time reservation; `armPressureJournal` returning
   * false is how a panel learns it is looking at a solver that cannot film.
   */
  get pressureJournalSource() { return this.resident.pressureJournalSource; }
  get pressureJournalLayout() { return this.resident.pressureJournalLayout; }
  get pressureJournalArmed() { return this.resident.pressureJournalArmed; }
  armPressureJournal(armed: boolean) { return this.resident.armPressureJournal(armed); }
  /** Header and iteration records of the last captured solve; maps a buffer. */
  readPressureJournal() { return this.resident.readPressureJournal(); }
  get globalFineLevelSetSource() { return this.resident.globalFineLevelSetSource; }

  private atlas: SparseAdaptiveMassAtlas;
  private lastTime_s = 0;
  private disposed = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = -Infinity;
  /** One undecodable hardware sample retires the chain for this solver. */
  private hardwarePhysicsTraceInvalid = false;

  private constructor(
    private readonly device: GPUDevice,
    // Not readonly: `applySceneUniforms` swaps in scalar-only scene revisions,
    // and `applyRuntimeValues` swaps the clock lane. Both are read fresh on
    // every advance rather than baked into an allocation, which is the whole
    // reason they can be adopted instead of rebuilt for.
    private scene: SceneDescription,
    private options: AdaptiveMassSolverOptions,
    private readonly presentation: WebGPUAdaptiveMassSparsePresentation,
    private readonly resident: WebGPUSparseCM12Resident,
    private readonly rigidSystem: WebGPURigidBodySystem | undefined,
    private readonly rigidExchange: GPUBuffer | undefined,
    private readonly rigidTerrainTexture: GPUTexture | undefined,
    private readonly rigidCouplingEnabled: boolean,
    adaptiveMixedSeamFaceCount: number,
    atlas: SparseAdaptiveMassAtlas,
    quality: GPUQuality,
  ) {
    this.atlas = atlas;
    this.volumeTexture = presentation.densityTexture;
    this.surfaceFieldTexture = presentation.levelSetTexture;
    this.gridCellTexture = presentation.gridCellTexture;
    this.velocityTexture = presentation.velocityTexture;
    this.gridPressureTexture = presentation.pressureTexture;
    this.gridDivergenceTexture = presentation.divergenceTexture;
    const stats = sparseBrickAtlasStats(atlas);
    const [nx, ny, nz] = atlas.dimensions;
    const representedFraction = stats.leafCount / Math.max(1, stats.equivalentFinestCellCount);
    const cellSize_m = Math.min(
      scene.container.width_m / nx,
      scene.container.height_m / ny,
      scene.container.depth_m / nz,
    );
    this.info = {
      nx,
      ny,
      nz,
      storedNy: ny,
      cellCount: stats.leafCount,
      equivalentUniformCells: stats.equivalentFinestCellCount,
      compressionRatio: representedFraction,
      activeCompressionRatio: representedFraction,
      activeSampleCount: stats.leafCount,
      regularLayers: ny,
      maximumNeighborDelta: 1,
      gridKind: "octree",
      cellSize_m,
      pressureIterations: 0,
      pressureSolver: "GPU-resident one-reduction composite GᵀWG sparse MGPCG",
      allocatedBytes: presentation.allocatedBytes + resident.allocatedBytes,
      quality,
      volumeCellSum: stats.integratedMassFineCells,
      representedVolumeCellSum: stats.integratedMassFineCells,
      representedVolumeDrift: 0,
      volumeTelemetrySource: "adaptive-conservative-mass",
      fluidBrickCapacity: stats.residentBrickCount,
      fluidBrickResidentCount: stats.residentBrickCount,
      fluidBrickCoreCount: stats.residentBrickCount,
      fluidBrickHaloCount: 0,
      fluidBrickGeneration: stats.generation,
      adaptiveFineBrickCount: stats.fineBrickCount,
      adaptiveCoarseBrickCount: stats.coarseBrickCount,
      adaptiveFineCoarseFaceConnectedPairCount:
        stats.fineCoarseFaceConnectedPairCount,
      adaptiveMixedSeamFaceCount,
      quadtreeMaximumFluidScale: 2,
      quadtreeMaximumNeighborRatio: 2,
      submittedTime_s: 0,
      simulatedTime_s: 0,
      completedTime_s: 0,
      simulationLag_s: 0,
      encodedSteps: 0,
      lastSubsteps: 1,
      maximumTallCellHeight: 2,
      surfaceField: "levelset",
      volumeControl: false,
      hostFluidAuthority: "gpu-resident",
      hostSimulationSizedWorkItems: 0,
      hostSchedulingUsesReadback: false,
    };
  }

  static async createAsync(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: AdaptiveMassSolverOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUAdaptiveMassSolver> {
    const runner = new GPUInitializationTaskRunner(onProgress, signal);
    // Compile the boundary chain's closing marker while the scene builds. A
    // recorder constructed before it exists closes on an empty pass, which
    // Metal never samples, and that one bad sample would retire hardware
    // timing for the whole run.
    void GPUStageTimestampRecorder.prepare(device);
    let dimensions: SparseBrickVec3 | undefined;
    let atlas: SparseAdaptiveMassAtlas | undefined;
    let presentation: WebGPUAdaptiveMassSparsePresentation | undefined;
    let grid: SparseAtlasCompositeGrid | undefined;
    let resident: WebGPUSparseCM12Resident | undefined;
    const rigidCouplingEnabled = scene.rigidBodies.length > 0;
    let rigidTerrainTexture: GPUTexture | undefined;
    let rigidExchange: GPUBuffer | undefined;
    let rigidSystem: WebGPURigidBodySystem | undefined;
    if (rigidCouplingEnabled) {
      const rigidDimensions = sceneLatticeDimensions(scene);
      const rigidCellHeight_m = Math.min(
        scene.container.width_m / rigidDimensions[0],
        scene.container.height_m / rigidDimensions[1],
        scene.container.depth_m / rigidDimensions[2],
      );
      const hasRigidTerrain = sceneHasTerrain(scene);
      const rigidTerrainWidth = hasRigidTerrain ? rigidDimensions[0] : 1;
      const rigidTerrainDepth = hasRigidTerrain ? rigidDimensions[2] : 1;
      rigidTerrainTexture = device.createTexture({
        label: "Sparse CM12 rigid terrain",
        size: [rigidTerrainWidth, rigidTerrainDepth],
        format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      if (hasRigidTerrain) {
        const terrain = terrainColumnHeights(scene, rigidTerrainWidth, rigidTerrainDepth);
        const rowBytes = rigidTerrainWidth * 4;
        const paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
        const terrainUpload = new Float32Array(paddedRowBytes / 4 * rigidTerrainDepth);
        for (let z = 0; z < rigidTerrainDepth; z += 1) {
          terrainUpload.set(Float32Array.from(
            terrain.slice(z * rigidTerrainWidth, (z + 1) * rigidTerrainWidth),
            (height) => height / rigidCellHeight_m,
          ), z * paddedRowBytes / 4);
        }
        device.queue.writeTexture(
          { texture: rigidTerrainTexture },
          terrainUpload,
          { bytesPerRow: paddedRowBytes, rowsPerImage: rigidTerrainDepth },
          [rigidTerrainWidth, rigidTerrainDepth],
        );
      }
      rigidExchange = device.createBuffer({
        label: "Sparse CM12 rigid exchange",
        size: GPU_RIGID_EXCHANGE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      rigidSystem = new WebGPURigidBodySystem(
        device, scene, rigidExchange, rigidTerrainTexture,
      );
      rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    }
    let initiallyActiveBrickKeys: ReadonlySet<number> | undefined;
    try {
      await runner.run([...(rigidSystem?.initializationTasks() ?? []), {
        id: "adaptive-mass.plan",
        phase: "planning",
        label: "Bound the arbitrary-scene presentation lattice",
        run: () => { dimensions = adaptiveMassPresentationDimensionsForScene(scene); },
      }, {
        id: "adaptive-mass.atlas",
        phase: "adaptive-topology",
        label: "Build resident 4³/8³ sparse bricks",
        dependencies: ["adaptive-mass.plan"],
        run: () => {
          const resolutionForBrick = options.resolutionMode === "all-fine"
            ? () => 8 as const
            : options.resolutionMode === "all-coarse"
              ? () => 4 as const
              : undefined;
          atlas = initializeSparseBrickAtlasFromScene(scene, {
            finestDimensions: dimensions!,
            surfaceFineRings: options.surfaceFineRings,
            ...(resolutionForBrick ? { resolutionForBrick } : {}),
          });
          const supported = residentSupportAtlas(atlas, options.resolutionMode);
          initiallyActiveBrickKeys = new Set(supported.bricks.map((brick) => brick.key));
          const receiverScale = adaptiveMassReceiverScaleForScene(
            scene, options.receiverSupportRings,
          );
          atlas = dormantReceiverDomain(supported, options.resolutionMode,
            receiverScale.supportRings, options.receiverFloor,
            receiverScale.minimumCapacityScale);
          // The runtime is GPU-resident from generation zero. Construct only
          // the topology oracle needed by the packer; the CPU dynamics state
          // used to allocate duplicate velocity, pressure, policy and
          // workspace graphs that were never stepped or returned.
          grid = buildSparseAtlasCompositeGrid(atlas);
        },
      }, {
        id: "adaptive-mass.presentation",
        phase: "allocation",
        label: "Allocate adaptive water and ownership textures",
        dependencies: ["adaptive-mass.atlas"],
        run: () => {
          presentation = new WebGPUAdaptiveMassSparsePresentation(device);
        },
      }, {
        id: "adaptive-mass.resident",
        phase: "allocation",
        label: "Pack compact GPU topology and allocate resident frame state",
        dependencies: ["adaptive-mass.atlas", "adaptive-mass.presentation"],
        run: async () => {
          resident = await WebGPUSparseCM12Resident.create(
            device, atlas!, grid!, finestCellSize(scene, atlas!),
            initiallyActiveBrickKeys,
            rigidCouplingEnabled ? {
              bodies: rigidSystem!.stateBuffer,
              exchange: rigidExchange!,
              worldDimensions_m: [scene.container.width_m, scene.container.height_m,
                scene.container.depth_m],
            } : undefined,
            // Sized from the iteration ceiling this solver was built with, so
            // the journal can hold the longest solve it will ever encode.
            options.pressureJournal
              ? { iterationCapacity: sparseCM12PressureIterations(
                options.pressureIterations) }
              : undefined,
          );
        },
      }, {
        id: "adaptive-mass.upload",
        phase: "upload",
        label: "Publish sparse atlas generation zero",
        dependencies: ["adaptive-mass.resident"],
        run: () => {
          const encoder = device.createCommandEncoder({
            label: "Sparse CM12 initial GPU publication",
          });
          resident!.encodeInitialPresentation(encoder, finestCellSize(scene, atlas!));
          device.queue.submit([encoder.finish()]);
        },
      }, {
        id: "adaptive-mass.warmup",
        phase: "warmup",
        label: "Fence adaptive presentation generation zero",
        dependencies: ["adaptive-mass.upload"],
        run: () => device.queue.onSubmittedWorkDone(),
      }]);
      return new WebGPUAdaptiveMassSolver(
        device, scene, options, presentation!, resident!,
        rigidSystem, rigidExchange, rigidTerrainTexture, rigidCouplingEnabled,
        grid!.mixedSeamRowCount, atlas!, quality,
      );
    } catch (error) {
      resident?.destroy();
      presentation?.destroy();
      rigidSystem?.destroy();
      rigidExchange?.destroy();
      rigidTerrainTexture?.destroy();
      throw error;
    }
  }

  /**
   * Add a ball of liquid to the atlas the running solve is stepping.
   *
   * Without this the editor authors the ball into the scene document instead,
   * which re-seeds this solver at t = 0 — the user drops water into a running
   * tank and the tank restarts. The ball is converted from metres to finest
   * cells here because the atlas is the only thing that knows the lattice, and
   * its radius becomes three radii: a metric sphere is an ellipsoid on any
   * lattice whose cells are not cubes.
   *
   * The drop is applied to the atlas immediately rather than deferred to the
   * next step, and the presentation is republished from it so the fields the
   * renderer and the diagnostics read agree with the atlas before that step.
   * The ball is drawn from the step after the drop, like any other water.
   */
  injectLiquidBall(ball: InjectedLiquidBall): void {
    if (this.disposed || !(ball.radius_m > 0)) return;
    const container = this.scene.container;
    const [nx, ny, nz] = this.atlas.dimensions;
    const encoder = this.device.createCommandEncoder({
      label: "Sparse CM12 GPU liquid injection",
    });
    this.resident.encodeLiquidInjection(
      encoder,
      finestCellSize(this.scene, this.atlas),
      [
        (ball.centre_m.x + 0.5 * container.width_m) * nx / container.width_m,
        ball.centre_m.y * ny / container.height_m,
        (ball.centre_m.z + 0.5 * container.depth_m) * nz / container.depth_m,
      ],
      [
        ball.radius_m * nx / container.width_m,
        ball.radius_m * ny / container.height_m,
        (ball.halfHeight_m ?? ball.radius_m) * nz / container.depth_m,
      ],
    );
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Adopt scene scalars on the running solver.
   *
   * Everything this method reads out of the document below — `numerics.maxDt_s`,
   * `fluid.gravity_m_s2`, `fluid.density_kg_m3` — is read per advance, never
   * baked into a buffer, a pipeline or an atlas. Without this the renderer had
   * no way to deliver a changed scalar except by constructing a new solver, so
   * nudging the step slider rebuilt the whole sparse world to arrive at an
   * identical one. The renderer only calls this once the structural and seed
   * tiers already match, so the incoming document differs in scalars alone.
   */
  applySceneUniforms(scene: SceneDescription): void {
    this.scene = scene;
  }

  /**
   * Adopt the controls that only change what the next advance asks for.
   *
   * `timeStep` picks between the paper 1/30 s operating step and the scene's
   * authored `maxDt_s`; both are consulted at the top of `advanceTo`, so the
   * switch is a live one. Activity thresholds are likewise copied into the
   * next frame's small policy uniform. Structural capacity controls still
   * rebuild, while accepted resolution changes publish at topology epochs.
   */
  applyRuntimeValues(values: MethodParamValues): void {
    const timeStep = values.timeStep === "scene" ? "scene" : "paper";
    const sharpeningDistance = sparseCM12SharpeningDistance(values.sharpeningDistance);
    const sharpeningTraceSteps = sparseCM12SharpeningTraceSteps(values.sharpeningTraceSteps);
    const pressureIterations = sparseCM12PressureIterations(values.pressureIterations);
    const pressureRelativeTolerance =
      sparseCM12PressureRelativeTolerance(values.pressureRelativeTolerance);
    const activityPolicy = sparseCM12ActivityPolicy({
      ...values,
      activitySignals: values.selectorMode === "activity",
    });
    this.options = { ...this.options, timeStep, sharpeningDistance, sharpeningTraceSteps,
      pressureIterations, pressureRelativeTolerance, activityPolicy };
  }

  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean {
    if (this.disposed || !Number.isFinite(time_s) || time_s <= this.lastTime_s + 1e-9) return false;
    const paperTimeStep = this.options.timeStep === "paper";
    if (paperTimeStep
      && time_s - this.lastTime_s < CM12_PAPER_DT_S - 1e-9) return false;
    const dt_s = paperTimeStep
      ? CM12_PAPER_DT_S
      : Math.min(this.scene.numerics.maxDt_s, time_s - this.lastTime_s);
    if (!(dt_s > 0)) return false;
    const cellSize_m = finestCellSize(this.scene, this.atlas);
    const activeBodies = bodies.slice(0, 12);
    this.rigidSystem?.syncBodies(activeBodies);
    const gravity = this.scene.fluid.gravity_m_s2;
    const instrumentation = usePerformanceInstrumentationStore.getState();
    const traceRequestedAt_ms = instrumentation.enabled ? performance.now() : 0;
    const shouldTracePhysics = instrumentation.enabled && !this.physicsTracePending
      && traceRequestedAt_ms - this.lastPhysicsTraceAt_ms
        >= ADAPTIVE_MASS_FRAME_TRACE_CADENCE_MS;
    const traceSampleId = shouldTracePhysics ? ++this.physicsTraceSampleId : 0;
    const traceContext = `adaptive-mass:sim-${(this.lastTime_s + dt_s).toFixed(6)}`;
    const frameCapture = shouldTracePhysics
      ? new AdaptiveMassFrameCapture(traceSampleId, traceContext)
      : undefined;
    const rawEncoder = this.device.createCommandEncoder({
      label: `Sparse CM12 resident frame ${(this.lastTime_s + dt_s).toFixed(6)}`,
    });
    // The stage partition is the encoder's own: boundaries ride the passes the
    // advance already encodes, so a sampled advance dispatches exactly the
    // physics an unsampled one does. `markersReady` gates the recorder's
    // fallback closing marker, which an advance whose final stage encoded
    // nothing would fall back to; an unsampled boundary there would retire
    // hardware timing for the whole run.
    const hardwareTrace = frameCapture && !this.hardwarePhysicsTraceInvalid
      && GPUStageTimestampRecorder.supported(this.device)
      && GPUStageTimestampRecorder.markersReady(this.device)
      ? new GPUStageTimestampRecorder(this.device, traceSampleId, "physics", traceContext)
      : undefined;
    const encoder = frameCapture
      ? frameCapture.instrument(rawEncoder, hardwareTrace)
      : rawEncoder;
    if (this.rigidExchange) encoder.clearBuffer(this.rigidExchange);
    this.resident.encode(
      encoder,
      dt_s,
      cellSize_m,
      this.scene.fluid.density_kg_m3 * cellSize_m * cellSize_m / dt_s,
      [gravity.x / cellSize_m, gravity.y / cellSize_m, gravity.z / cellSize_m],
      {
        distanceCells: this.options.sharpeningDistance,
        traceSteps: this.options.sharpeningTraceSteps,
      },
      this.options.activityPolicy,
      {
        iterations: this.options.pressureIterations,
        relativeTolerance: this.options.pressureRelativeTolerance,
      },
      frameCapture?.residentStageSeams,
      this.rigidCouplingEnabled ? activeBodies.length : 0,
      [this.scene.container.width_m, this.scene.container.height_m,
        this.scene.container.depth_m],
    );
    this.rigidSystem?.encode(encoder, dt_s, cellSize_m ** 3, 1, cellSize_m);
    frameCapture?.closeCommands();
    this.device.queue.submit([encoder.finish()]);

    this.lastTime_s += dt_s;
    const nextTime_s = this.lastTime_s;
    this.info.submittedTime_s = nextTime_s;
    this.info.simulatedTime_s = nextTime_s;
    this.info.simulationLag_s = Math.max(0, time_s - nextTime_s);
    this.info.lastDt_s = dt_s;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    this.info.lastSubsteps = 1;
    this.info.pressureIterations = sparseCM12PressureIterations(
      this.options.pressureIterations);
    this.info.hostSimulationSizedWorkItems = 0;
    const captured = frameCapture?.finish(this.device.queue);
    this.finishFrameCapture(captured, traceRequestedAt_ms);
    return true;
  }

  private finishFrameCapture(
    captured: ReturnType<AdaptiveMassFrameCapture["finish"]> | undefined,
    traceRequestedAt_ms: number,
  ): void {
    if (captured) {
      this.lastPhysicsTraceAt_ms = traceRequestedAt_ms;
      this.physicsTracePending = true;
      this.info.physicsCPUTrace = captured.cpuTrace;
      this.info.physicsCaptureIdentity = captured.identity;
      // Prefer the hardware partition: it is the only lane that can put a
      // figure on an individual stage. One unusable sample retires it for this
      // solver and the queue-wall observation carries the advance from then on.
      const resolved = captured.hardwareTrace
        ? captured.hardwareTrace.then((trace) => {
          this.hardwarePhysicsTraceInvalid = !trace;
          return trace ?? captured.queueTrace;
        }).catch(() => {
          this.hardwarePhysicsTraceInvalid = true;
          return captured.queueTrace;
        })
        : captured.queueTrace;
      void Promise.resolve(resolved).then((trace) => {
        const current = usePerformanceInstrumentationStore.getState();
        if (trace && !this.disposed && current.enabled
          && current.enabledAt_ms <= traceRequestedAt_ms) {
          this.info.physicsTrace = trace;
        }
      }).catch(() => {}).finally(() => {
        this.physicsTracePending = false;
      });
    }
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.device.queue.onSubmittedWorkDone();
    const diagnostics = await this.resident.readDiagnostics();
    // Telemetry readback is deliberately downstream of simulation. These
    // values update panels only; no scheduler decision or dispatch dimension
    // is ever derived from them on the host.
    const topology = diagnostics as typeof diagnostics
      & SparseCM12TopologySchedulerDiagnostics;
    this.info.pressureRelativeResidual = diagnostics.pressureRelativeResidual;
    this.info.pressureRecursiveRelativeResidual =
      diagnostics.pressureRecursiveRelativeResidual;
    this.info.pressureTrueResidualMaximum = diagnostics.pressureTrueResidualMaximum;
    this.info.pressureInitialTrueRelativeResidual =
      diagnostics.pressureInitialTrueRelativeResidual;
    this.info.pressureIterationsExecuted = diagnostics.pressureIterationsExecuted;
    this.info.pressureIterationsEncoded = diagnostics.pressureIterationsEncoded;
    this.info.pressureFirstToleranceCrossingIteration =
      diagnostics.pressureFirstToleranceCrossingIteration;
    this.info.pressureSolveConverged = diagnostics.pressureSolveConverged;
    this.info.pressureIterationCapReached = diagnostics.pressureIterationCapReached;
    this.info.pressureConvergenceReason = diagnostics.pressureConvergenceReason;
    this.info.pressureCurvatureBreakdown = diagnostics.pressureCurvatureBreakdown;
    this.info.pressureCurvatureRecoveryCount =
      diagnostics.pressureCurvatureRecoveryCount;
    this.info.pressureRecursiveToTrueResidualRatio =
      diagnostics.pressureRecursiveToTrueResidualRatio;
    this.info.pressureResidualDrift = diagnostics.pressureResidualDrift;
    this.info.maxDivergenceAfter_s = diagnostics.maximumDivergence_s;
    this.info.maxDivergence_s = diagnostics.maximumDivergence_s;
    const adaptiveInfo = this.info as typeof this.info & AdaptiveMassStepTelemetry;
    adaptiveInfo.adaptiveMaximumMixedSeamDivergence_s =
      diagnostics.maximumMixedSeamDivergence_s;
    adaptiveInfo.adaptiveMaximumInactiveFaceSpeedAfter_m_s = 0;
    this.info.adaptiveActivityMaximumScore = diagnostics.activityMaximumScore;
    this.info.adaptiveActivityMeasuredBrickCount = diagnostics.activityMeasuredBrickCount;
    this.info.adaptiveActivitySurfaceBrickCount = diagnostics.activitySurfaceBrickCount;
    this.info.adaptiveActivityHotBrickCount = diagnostics.activityHotBrickCount;
    this.info.adaptiveActivityQuietBrickCount = diagnostics.activityQuietBrickCount;
    this.info.adaptiveResolutionTopologyEpoch = diagnostics.activityTopologyEpoch;
    this.info.activeSampleCount = diagnostics.acceptedCellCount;
    this.info.activeCompressionRatio = diagnostics.acceptedCellCount
      / Math.max(1, this.info.equivalentUniformCells ?? diagnostics.acceptedCellCount);
    this.info.fluidBrickResidentCount = diagnostics.activeBrickCount;
    this.info.fluidBrickCoreCount = diagnostics.activeBrickCount;
    // Residency and accepted split/merge publication are independent GPU
    // generations. Their sum is a monotonic renderer-facing revision.
    this.info.fluidBrickGeneration = this.atlas.generation
      + diagnostics.residencyGeneration + (topology.acceptedTopologyGeneration ?? 0);
    this.info.adaptiveTopologyShadowGeneration =
      topology.acceptedTopologyGeneration ?? 0;
    this.info.adaptiveTopologyUrgentQueuedBrickCount =
      topology.topologyUrgentQueuedBrickCount ?? 0;
    this.info.adaptiveTopologyOrdinaryQueuedBrickCount =
      topology.topologyOrdinaryQueuedBrickCount ?? 0;
    this.info.adaptiveTopologyPreparedBrickCount = topology.topologyPreparedBrickCount ?? 0;
    this.info.adaptiveTopologyCommittedBrickCount = topology.topologyCommittedBrickCount ?? 0;
    this.info.adaptiveTopologyDeferredBrickCount = topology.topologyDeferredBrickCount ?? 0;
    this.info.adaptiveTopologyShadowFineBrickCount = topology.acceptedFineBrickCount;
    this.info.adaptiveTopologyShadowCoarseBrickCount = topology.acceptedCoarseBrickCount;
    this.info.adaptiveAcceptedCellCount = diagnostics.acceptedCellCount;
    this.info.adaptiveAcceptedRowCount = diagnostics.acceptedRowCount;
    this.info.adaptiveAcceptedSameLevelCoarseRowCount =
      diagnostics.acceptedSameLevelCoarseRowCount;
    this.info.adaptiveAcceptedMixedSeamRowCount = diagnostics.acceptedMixedSeamRowCount;
    this.info.adaptivePressureActiveRowCount = diagnostics.pressureActiveRowCount;
    this.info.adaptivePressureCellCount = diagnostics.pressureCellCount;
    // Keep the established diagnostics/benchmark field live while callers
    // migrate to the pressure-specific name above.
    this.info.adaptiveMixedSeamFaceCount = diagnostics.acceptedMixedSeamRowCount;
    this.info.adaptiveFineBrickCount = topology.acceptedFineBrickCount;
    this.info.adaptiveCoarseBrickCount = topology.acceptedCoarseBrickCount;
    this.info.adaptiveResolutionPromotedBrickCount = 0;
    this.info.adaptiveResolutionDemotedBrickCount = 0;
    this.info.adaptiveResolutionDeferredPromotionCount = 0;
    this.info.completedTime_s = Math.max(
      this.info.completedTime_s ?? 0,
      this.info.submittedTime_s ?? 0,
    );
    return { ...this.info };
  }

  /** Explicit Dawn/QA materialization; production rendering stays sparse. */
  readDiagnosticFields() { return this.resident.readDiagnosticFields(); }

  get rigidRenderBuffer(): GPUBuffer | undefined { return this.rigidSystem?.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer | undefined { return this.rigidSystem?.motionBuffer; }
  setSelectedRigidBody(index: number): void { this.rigidSystem?.setSelectedIndex(index); }
  async pickRigidBody(origin: RigidBodyState["position_m"],
    direction: RigidBodyState["position_m"]) {
    return this.rigidSystem?.pick(origin, direction);
  }
  async readRigidBodyPoses() { return this.rigidSystem?.readPoses(); }

  /** Explicit acceptance/debug readback; never consulted by advanceTo. */
  async readGPUActivityPolicy(): Promise<{
    readonly acceptedSteps: number;
    readonly bricks: readonly AdaptiveMassGPUActivityBrick[];
  }> {
    const snapshot = await this.resident.readActivitySnapshot();
    if (snapshot.records.length !== this.atlas.bricks.length) {
      throw new Error("Sparse CM12 GPU activity record count does not match resident bricks");
    }
    return {
      acceptedSteps: snapshot.acceptedSteps,
      bricks: snapshot.records.map((record, index) => {
        const brick = this.atlas.bricks[index]!;
        return { ...record, key: brick.key, coordinate: brick.coordinate,
          resolution: brick.resolution };
      }),
    };
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.resident.destroy();
    this.presentation.destroy();
    this.rigidSystem?.destroy();
    this.rigidExchange?.destroy();
    this.rigidTerrainTexture?.destroy();
  }
}

export function adaptiveMassPresentationDimensionsForScene(
  scene: SceneDescription,
): SparseBrickVec3 {
  return sceneLatticeDimensions(scene);
}

function finestCellSize(scene: SceneDescription, atlas: SparseAdaptiveMassAtlas): number {
  return Math.min(
    scene.container.width_m / atlas.dimensions[0],
    scene.container.height_m / atlas.dimensions[1],
    scene.container.depth_m / atlas.dimensions[2],
  );
}
