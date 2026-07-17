import { adaptiveOpticalLayerDefaults, adaptivePressureCellTopology, buildAdaptiveOpticalLayerField, buildQuadtree, buildVariationalSystem, maximumVelocityUpdateFluidScale, populateTallPressureGrid, populateTallPressureGridFromLeafProfiles, quadtreeFromPackedCells, quadtreeSizingFromVelocityAndSurface, signedDistanceFromVolume, type AdaptiveOpticalLayerField, type QuadtreeGrid, type TallPressureGrid, type TallPressureSample, type VariationalBody, type VariationalSystem } from "./quadtree-tall-cell-grid";
import { damBreakFractions } from "./initial-fluid";
import { insidePrimitive } from "./fluid-rigid-coupling";
import { boundingRadius, quaternionRotate, type RigidBodyState } from "./rigid-body";
import type { SceneDescription, Vec3 } from "./model";
import { WebGPUQuadtreeBuilder, WebGPUQuadtreeSurfaceState, type SurfaceInflowState, type WebGPUQuadtreeConstructionCache, type WebGPUQuadtreeSurfaceCache } from "./webgpu-quadtree-builder";
import { createInflowGridBoundary, inflowBoundaryWGSL, inflowOutletCenter } from "./inflow-boundary";
import { WebGPUQuadtreePackBuilder, type GPUQuadtreeResidentResources } from "./webgpu-quadtree-pack-builder";
import { sceneHasTerrain, terrainCellSolidFraction, terrainColumnHeights } from "./terrain";
import { buildQuadtreeMultigridHierarchy, type QuadtreeMultigridHierarchy } from "./quadtree-multigrid";

export interface QuadtreeRigidCoupling {
  bodies: RigidBodyState[];
  /** True when a load consumer integrates the bodies; kinematic bodies keep M^-1 = 0. */
  dynamic: boolean;
}

/**
 * The WGSL, bind-group layout, and compiled pipelines are identical for every
 * rebuilt projection; recompiling them per topology rebuild costs tens of
 * milliseconds per step. The first projection fills this cache and every
 * rebuild reuses it.
 */
export interface QuadtreeGPUCache {
  layout: GPUBindGroupLayout;
  module: GPUShaderModule;
  pipelineLayout: GPUPipelineLayout;
  pipelines?: Record<string, GPUComputePipeline>;
  dispatchLayout?: GPUBindGroupLayout;
  dispatchPipeline?: GPUComputePipeline;
  velocityExtrapolationLayout?: GPUBindGroupLayout;
  velocityExtrapolationPipeline?: GPUComputePipeline;
  divergenceLayout?: GPUBindGroupLayout;
  divergencePipeline?: GPUComputePipeline;
  velocityClampLayout?: GPUBindGroupLayout;
  velocityClampPipeline?: GPUComputePipeline;
  surfaceTransportLayout?: GPUBindGroupLayout;
  surfaceTransportPipeline?: GPUComputePipeline;
  construction?: WebGPUQuadtreeConstructionCache;
  surface?: WebGPUQuadtreeSurfaceCache;
  cpuWorker?: { postMessage(message: unknown, transfer?: ArrayBuffer[]): void; terminate(): unknown };
  cpuWorkerSequence?: number;
  cpuWorkerPending?: Map<number, { resolve: (value: PreparedProjectionCPU) => void; reject: (reason: unknown) => void }>;
  gpuPack?: WebGPUQuadtreePackBuilder;
  multigridLayout?: GPUBindGroupLayout;
  multigridModule?: GPUShaderModule;
  multigridPipelineLayout?: GPUPipelineLayout;
  multigridPipelines?: Record<string, GPUComputePipeline>;
}

export interface QuadtreeBodyImpulse {
  bodyId: string;
  impulse_N_s: Vec3;
  angularImpulse_N_m_s: Vec3;
  displacedVolume_m3: number;
}

function solidFieldsFromBodies(scene: SceneDescription, bodies: RigidBodyState[], nx: number, ny: number, nz: number, h: Vec3) {
  const hasTerrain = sceneHasTerrain(scene);
  if (bodies.length === 0 && !hasTerrain) return undefined;
  const solidFraction = new Float32Array(nx * ny * nz);
  const solidOwner = new Int32Array(nx * ny * nz).fill(-1);
  const halfWidth = scene.container.width_m / 2, halfDepth = scene.container.depth_m / 2;
  // Terrain is a static solid with zero velocity: owner stays -1, so the
  // variational A u_fluid + (1-A) u_solid face flux gets u_solid = 0 and no
  // rigid-coupling row, exactly a no-slip ground at the column height.
  if (hasTerrain) {
    const heights = terrainColumnHeights(scene, nx, nz);
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const height = heights[x + nx * z];
      for (let y = 0; y < ny; y += 1) {
        const fraction = terrainCellSolidFraction(height, y * h.y, h.y);
        if (fraction <= 0) break;
        solidFraction[x + nx * (y + ny * z)] = fraction;
      }
    }
  }
  // A display nozzle is a filled rigid primitive; its open channel is the
  // prescribed inflow cylinder, which must stay carved out of [A] exactly as
  // the legacy coupling kernel carved its inflow velocity cells.
  const inflow = scene.fluid.inflow;
  const inflowSpeed = inflow ? Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z) : 0;
  const inflowDirection = inflow && inflowSpeed > 0 ? { x: inflow.velocity_m_s.x / inflowSpeed, y: inflow.velocity_m_s.y / inflowSpeed, z: inflow.velocity_m_s.z / inflowSpeed } : undefined;
  const margin = Math.max(h.x, h.y, h.z);
  const insideInflowChannel = (point: Vec3) => {
    if (!inflow || !inflowDirection) return false;
    const dx = point.x - inflow.center_m.x, dy = point.y - inflow.center_m.y, dz = point.z - inflow.center_m.z;
    const along = dx * inflowDirection.x + dy * inflowDirection.y + dz * inflowDirection.z;
    if (Math.abs(along) > inflow.length_m / 2 + margin) return false;
    const rx = dx - along * inflowDirection.x, ry = dy - along * inflowDirection.y, rz = dz - along * inflowDirection.z;
    return Math.hypot(rx, ry, rz) <= inflow.radius_m + margin;
  };
  bodies.forEach((body, owner) => {
    const radius = boundingRadius(body) + Math.max(h.x, h.y, h.z);
    const localX = body.position_m.x + halfWidth, localY = body.position_m.y, localZ = body.position_m.z + halfDepth;
    const x0 = Math.max(0, Math.floor((localX - radius) / h.x)), x1 = Math.min(nx - 1, Math.ceil((localX + radius) / h.x));
    const y0 = Math.max(0, Math.floor((localY - radius) / h.y)), y1 = Math.min(ny - 1, Math.ceil((localY + radius) / h.y));
    const z0 = Math.max(0, Math.floor((localZ - radius) / h.z)), z1 = Math.min(nz - 1, Math.ceil((localZ + radius) / h.z));
    for (let z = z0; z <= z1; z += 1) for (let y = y0; y <= y1; y += 1) for (let x = x0; x <= x1; x += 1) {
      let inside = 0;
      for (let corner = 0; corner < 8; corner += 1) {
        const point = {
          x: -halfWidth + (x + 0.5 + ((corner & 1) > 0 ? 0.4 : -0.4)) * h.x,
          y: (y + 0.5 + ((corner & 2) > 0 ? 0.4 : -0.4)) * h.y,
          z: -halfDepth + (z + 0.5 + ((corner & 4) > 0 ? 0.4 : -0.4)) * h.z
        };
        if (insidePrimitive(body, point)) inside += 1;
      }
      if (inside === 0) continue;
      if (insideInflowChannel({ x: -halfWidth + (x + 0.5) * h.x, y: (y + 0.5) * h.y, z: -halfDepth + (z + 0.5) * h.z })) continue;
      const index = x + nx * (y + ny * z), fraction = inside / 8;
      if (fraction > solidFraction[index]) { solidFraction[index] = fraction; solidOwner[index] = owner; }
    }
  });
  return { solidFraction, solidOwner };
}

function variationalBodiesFrom(scene: SceneDescription, coupling: QuadtreeRigidCoupling): VariationalBody[] {
  const rho = scene.fluid.density_kg_m3, halfWidth = scene.container.width_m / 2, halfDepth = scene.container.depth_m / 2;
  return coupling.bodies.map((body) => {
    let inverseMass = 0;
    const inverseInertia = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    if (coupling.dynamic) {
      inverseMass = rho * body.inverseMass_kg;
      const q = body.orientation;
      const columns = [quaternionRotate(q, { x: 1, y: 0, z: 0 }), quaternionRotate(q, { x: 0, y: 1, z: 0 }), quaternionRotate(q, { x: 0, y: 0, z: 1 })];
      const invBody = [body.inverseInertiaBody_kg_m2.x, body.inverseInertiaBody_kg_m2.y, body.inverseInertiaBody_kg_m2.z];
      const axes = ["x", "y", "z"] as const;
      for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
        let sum = 0;
        for (let k = 0; k < 3; k += 1) sum += columns[k][axes[row]] * invBody[k] * columns[k][axes[column]];
        inverseInertia[3 * row + column] = rho * sum;
      }
    }
    return {
      position: { x: body.position_m.x + halfWidth, y: body.position_m.y, z: body.position_m.z + halfDepth },
      linearVelocity: body.linearVelocity_m_s,
      angularVelocity: body.angularVelocity_rad_s,
      inverseMass, inverseInertia
    };
  });
}

const INVALID = 0xffffffff;

export interface QuadtreeTallCellProjectionOptions {
  pressureIterations: number;
  relativeTolerance: number;
  adaptivityStrength: number;
  maximumLeafSize: number;
  opticalDepthFraction: number;
  /** Fixed quarter-depth baseline or Narita--Kanai 2026 motion-adaptive layer. */
  opticalLayerMode?: "fixed" | "adaptive-motion";
  /** Eq. (1) motion-error scale; the paper uses 0.5. */
  opticalAlpha?: number;
  /**
   * Scale on the deep-interior speed-gradient sizing term (1 = paper-faithful
   * column reduction). Below 1, fast but smooth bulk flow stops forcing the
   * finest leaves far from the surface; the near-surface band keeps the full
   * curvature/strain/speed/front-speed formula.
   */
  deepSpeedGradientScale?: number;
  /** IC(0) is paper-conformant; the other choices preserve the operator and tolerance stop. */
  preconditioner?: "ic0" | "blockic" | "jacobi" | "line" | "poly" | "mg";
  /** Degree of the damped-Jacobi Neumann polynomial (2--4). */
  polynomialDegree?: number;
  /**
   * Row-parallel Chebyshev is the throughput path. PCG retains the exact
   * tolerance-driven ladder and monolithic rigid response for A/B checks.
   */
  pressureSolver?: QuadtreePressureSolver;
  /** Internal feedback carried across topology rebuilds. */
  iterationBudgetHint?: number;
  /** Internal exponential moving average of iterations-to-tolerance. */
  iterationEmaHint?: number;
  /** Internal condition-number guard for very tall cells created by thin optical layers. */
  iterationConditioningScale?: number;
  /** Opt-in timestamp-query breakdown of setup / early iterations / remainder / projection. */
  debugPressureTimings?: boolean;
  debugPressureFirstIterations?: number;
  /** Non-paper isolated-voxel hygiene; opt-in only. */
  debrisCulling?: boolean;
  /** Catastrophic lost-liquid safety circuit; inactive during healthy phi transport. */
  vofReconciliation?: boolean;
  /**
   * Seed each solve from the previous step's mapped cubical pressure instead
   * of zero (poly/jacobi/multigrid paths). The stop test stays relative to |b|, so a
   * converged answer is identical to the cold start within tolerance; steps
   * whose warm residual already meets tolerance skip the CG loop entirely.
   */
  pressureWarmStart?: boolean;
  /** Legacy master switch; false always disables the persistent solver. */
  megakernelSolve?: boolean;
  /** Dynamic is measured/default, always forces supported solves, and off disables it. */
  megakernelMode?: QuadtreeMegakernelMode;
  /** Maximum DOFs considered by dynamic megakernel selection. */
  megakernelDofLimit?: number;
  /** Maximum effective row-iterations considered by dynamic selection. */
  megakernelRowIterationLimit?: number;
  /** Internal: iterations used by the last converged solve, carried across rebuilds. */
  megakernelIterationHint?: number;
}

export interface QuadtreeIterationBudget {
  hardBudget: number;
  encodedBudget: number;
  ema: number;
}

/** Pure helper kept separate from GPU state so budget feedback is unit-testable. */
export function quadtreeIterationBudget(dofCount: number, options: Pick<QuadtreeTallCellProjectionOptions, "pressureIterations" | "iterationBudgetHint" | "iterationEmaHint" | "iterationConditioningScale">): QuadtreeIterationBudget {
  const hardBudget = Math.max(options.pressureIterations, Math.min(2048, Math.ceil(4 * Math.sqrt(Math.max(1, dofCount) * Math.max(1, options.iterationConditioningScale ?? 1)))));
  const ema = Math.max(1, options.iterationEmaHint ?? options.iterationBudgetHint ?? hardBudget / 1.5);
  const hinted = options.iterationBudgetHint ?? Math.ceil(1.5 * ema);
  return { hardBudget, encodedBudget: Math.max(Math.min(8, hardBudget), Math.min(hardBudget, Math.ceil(hinted))), ema };
}

export function nextQuadtreeIterationBudget(current: QuadtreeIterationBudget, used: number, converged: boolean): QuadtreeIterationBudget {
  const boundedUsed = Math.max(0, Math.min(current.hardBudget, Math.round(used)));
  // Asymmetric tracking: follow drops quickly (warm-started calm scenes reach
  // tolerance in a handful of iterations and every over-encoded iteration is
  // paid as dispatch decode), climb cautiously and rely on the immediate 2x
  // recovery below when a solve exhausts its budget.
  const alpha = boundedUsed < current.ema ? 0.4 : 0.25;
  const ema = current.ema * (1 - alpha) + boundedUsed * alpha;
  const target = !converged && boundedUsed >= current.encodedBudget
    ? Math.max(current.encodedBudget + 1, current.encodedBudget * 2)
    // Indirect work becomes zero after convergence, but WebGPU still decodes
    // every pre-encoded dispatch at near-full per-dispatch cost, so the tail
    // is priced like live iterations. Keep 20% variation headroom plus a
    // small absolute margin and rely on the immediate 2x recovery when a
    // solve reaches this budget. The low floor matters for settled scenes,
    // where the solve converges in a handful of iterations and a large floor
    // would encode a permanently dead tail.
    : 1.2 * ema + 4;
  return {
    hardBudget: current.hardBudget,
    encodedBudget: Math.max(Math.min(8, current.hardBudget), Math.min(current.hardBudget, Math.ceil(target))),
    ema
  };
}

export const quadtreeMegakernelDofLimit = 32_768;
export const quadtreeMegakernelRowIterationLimit = 30_000;
export type QuadtreeMegakernelMode = "dynamic" | "always" | "off";
export type QuadtreePressureSolver = "chebyshev" | "pcg";

/** Match the octree's measured pressure-effort policy. */
export function quadtreeChebyshevPasses(pressureIterations: number) {
  return Math.max(1, Math.ceil(Math.max(1, pressureIterations) / 4));
}

/**
 * The persistent kernel serializes the whole matrix onto one workgroup. It is
 * profitable for already-calm warm solves, but parallel ladder dispatches win
 * once row count times observed iterations grows. Require one converged
 * ladder observation before opting in, and charge higher polynomial degrees
 * for their extra matrix products.
 */
export function quadtreeMegakernelPreferred(
  dofCount: number,
  previousIterations: number | undefined,
  polynomialDegree = 2,
  dofLimit = quadtreeMegakernelDofLimit,
  rowIterationLimit = quadtreeMegakernelRowIterationLimit
) {
  if (!Number.isFinite(dofCount) || dofCount < 0 || previousIterations === undefined
    || !Number.isFinite(previousIterations) || previousIterations < 0
    || !Number.isFinite(dofLimit) || dofLimit < 0
    || !Number.isFinite(rowIterationLimit) || rowIterationLimit < 0) return false;
  const degreeCost = Math.max(1, Math.max(1, polynomialDegree) / 2);
  const effectiveRowIterations = dofCount * Math.max(1, Math.round(previousIterations)) * degreeCost;
  return dofCount <= dofLimit && effectiveRowIterations <= rowIterationLimit;
}

export interface QuadtreeTallCellProjectionInfo {
  leafCount: number;
  pressureSampleCount: number;
  liquidDofCount: number;
  faceCount: number;
  mlsProjectionRowCount: number;
  tallSegmentCount: number;
  ghostFaceCount: number;
  maximumNeighborRatio: number;
  compressionRatio: number;
  maximumFluidScale: number;
  opticalLayerMode: "fixed" | "adaptive-motion";
  opticalAlpha: number;
  opticalMinimumCells: number;
  opticalMaximumCells: number;
  allocatedBytes: number;
  /** GPU queue + compact readback time for Sec. 4.1 construction. */
  gpuConstruction_ms?: number;
  /** Timestamp-query duration of only the GPU sizing/subdivision kernels. */
  gpuConstructionKernel_ms?: number;
  /** GPU wall time for Sec. 4.2 segmentation plus Sec. 4.4 face/CSR emission. */
  gpuSparsePack_ms?: number;
  /** CPU time left for tall-cell/variational sparse packing after the tree exists. */
  cpuTopologyPack_ms?: number;
  cpuRedistance_ms?: number;
  cpuQuadtreeDecode_ms?: number;
  cpuTallGrid_ms?: number;
  cpuVariationalAssembly_ms?: number;
  cpuSystemPack_ms?: number;
  cpuICFactorization_ms?: number;
  cpuResourceUpload_ms?: number;
  topologyReused?: boolean;
  topologyReuseCount?: number;
  pressureIterationsUsed?: number;
  pressureIterationBudget?: number;
  pressureIterationHardBudget?: number;
  pressureConverged?: boolean;
  factorLevelCount?: number;
  multigridLevelCount?: number;
  multigridCoarsestDofs?: number;
  velocityClampCount?: number;
  pressurePhaseTimings?: { setup_ms: number; firstIterations_ms: number; remainingIterations_ms: number; project_ms: number };
  /** Bytes read back for an update (leaf-centre phi profiles + compact 2D leaves + diagnostics). */
  topologyReadbackBytes?: number;
}

interface ProjectionResources {
  velocityIn: GPUTexture;
  velocityOut: GPUTexture;
  /** Reused advection scratch; projection does not own its lifetime. */
  velocityScratch: GPUTexture;
  volume: GPUTexture;
  levelSet?: WebGPUQuadtreeSurfaceState;
  /** Shared flux-consistent velocity view consumed only by the surface pipeline. */
  surfaceTransport?: GPUTexture;
}

interface ProjectionFields {
  phi?: Float32Array;
  velocity?: Vec3[];
  quadtree?: QuadtreeGrid;
  pressureGrid?: TallPressureGrid;
  topologyWords?: Uint32Array;
  prepared?: PreparedProjectionCPU;
}

export interface QuadtreeCPUPreparationInput {
  scene: SceneDescription;
  dims: { nx: number; ny: number; nz: number };
  options: QuadtreeTallCellProjectionOptions;
  packedCells: Uint32Array;
  columnProfiles: Float32Array;
  opticalColumns?: Uint32Array;
  /** Current resident topology; matching rebuilds skip sparse/MLS repacking. */
  reuseTopologyWords?: Uint32Array;
  coupling?: QuadtreeRigidCoupling;
}

export interface PreparedProjectionCPU {
  quadtree?: QuadtreeGrid;
  pressureGrid?: TallPressureGrid;
  /** Compact topology metadata retained after the worker discards its graph objects. */
  leafCount: number;
  pressureSampleCount: number;
  maximumNeighborRatio: number;
  topologyWords: Uint32Array;
  packed?: ReturnType<typeof WebGPUQuadtreeTallCellProjection.packSystem>;
  /** GPU-owned sparse resources from W6's resident count-scan-emit path. */
  resident?: GPUQuadtreeResidentResources;
  reusedTopology?: boolean;
  /** False when topologyWords is only the horizontal owner map, not the full pressure segmentation identity. */
  topologyIdentityComplete?: boolean;
  gpuPack_ms?: number;
  displacedVolumes: number[];
  dofCount: number;
  faceCount: number;
  ghostFaceCount: number;
  maximumFluidScale: number;
  tallSegmentCount: number;
  quadtreeDecode_ms: number;
  tallGrid_ms: number;
  variationalAssembly_ms: number;
  systemPack_ms: number;
  icFactorization_ms: number;
}

/** Zero-copy worker handoff for the large immutable projection pack. */
export function preparedProjectionTransferables(value: PreparedProjectionCPU): ArrayBuffer[] {
  const buffers = new Set<ArrayBuffer>();
  const add = (view: ArrayBufferView | undefined) => { if (view?.buffer instanceof ArrayBuffer) buffers.add(view.buffer); };
  add(value.topologyWords);
  const packed = value.packed;
  if (packed) for (const view of [
    packed.faces, packed.rowOffsets, packed.rowEntries, packed.matrixWords, packed.cellProjection, packed.cellTopology,
    packed.factorColumns, packed.factorEntries, packed.factorAuxWords, packed.cellPressureSamples
  ]) add(view);
  for (const level of packed?.multigrid?.levels ?? []) {
    add(level.rowOffsets); add(level.columns); add(level.lineOffsets); add(level.lineNodes);
    add(level.nodeToCoarse); add(level.entryToCoarse); add(level.coarseNodeOffsets); add(level.coarseNodes); add(level.matrixWords);
  }
  return [...buffers];
}

function pressureTopologyWords(grid: TallPressureGrid) {
  const words = new Uint32Array(3 + grid.quadtree.leaves.length + 2 * grid.segments.length);
  words.set([grid.quadtree.leaves.length, grid.samples.length, grid.segments.length]);
  let cursor = 3;
  for (const leaf of grid.quadtree.leaves) words[cursor++] = leaf.x | (leaf.z << 10) | (leaf.size << 20);
  for (const segment of grid.segments) {
    const bottom = grid.samples[segment.bottomSample], top = grid.samples[segment.topSample];
    words[cursor++] = segment.leaf;
    words[cursor++] = segment.firstY | (segment.lastY << 10) | ((segment.tall ? 1 : 0) << 20) | ((bottom.liquid ? 1 : 0) << 21) | ((top.liquid ? 1 : 0) << 22);
  }
  return words;
}

function sameWords(a: Uint32Array, b: Uint32Array) {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) if (a[index] !== b[index]) return false;
  return true;
}

function displacedVolumesForGrid(grid: TallPressureGrid, phi: Float32Array | undefined, solidFields: ReturnType<typeof solidFieldsFromBodies>, bodyCount: number, nx: number, ny: number, h: Vec3) {
  const volumes = Array.from({ length: bodyCount }, () => 0);
  if (!solidFields) return volumes;
  const liquidByLeafY = phi ? undefined : new Uint8Array(grid.quadtree.leaves.length * ny);
  if (liquidByLeafY) for (const segment of grid.segments) {
    const liquid = grid.samples[segment.bottomSample].liquid ? 1 : 0;
    for (let y = segment.firstY; y <= segment.lastY; y += 1) liquidByLeafY[segment.leaf * ny + y] = liquid;
  }
  const cellVolume = h.x * h.y * h.z;
  for (let index = 0; index < solidFields.solidFraction.length; index += 1) {
    const owner = solidFields.solidOwner[index];
    if (owner < 0) continue;
    const x = index % nx, y = Math.floor(index / nx) % ny, z = Math.floor(index / (nx * ny));
    const liquid = phi ? phi[index] < 0 : liquidByLeafY![grid.quadtree.leafAt[x + nx * z] * ny + y] !== 0;
    if (liquid) volumes[owner] += solidFields.solidFraction[index] * cellVolume;
  }
  return volumes;
}

function initialFields(scene: SceneDescription, nx: number, ny: number, nz: number) {
  const count = nx * ny * nz, phi = new Float32Array(count), velocity = Array.from({ length: count }, () => ({ x: 0, y: 0, z: 0 }));
  const dam = damBreakFractions(scene.container.fillFraction);
  const heights = sceneHasTerrain(scene) ? terrainColumnHeights(scene, nx, nz) : undefined;
  const cellHeight = scene.container.height_m / ny;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const aboveGround = !heights || (y + 0.5) * cellHeight > heights[x + nx * z];
    const wet = aboveGround && (scene.fluid.initialCondition === "tank-fill"
      ? (y + 0.5) / ny <= scene.container.fillFraction
      : (x + 0.5) / nx <= dam.width && (y + 0.5) / ny <= dam.height && (z + 0.5) / nz <= dam.depth);
    phi[x + nx * (y + ny * z)] = wet ? -1 : 1;
  }
  return { phi, velocity };
}

function initialSizing(scene: SceneDescription, nx: number, nz: number, h: Vec3, bodies?: RigidBodyState[]) {
  const sizing = new Float32Array(nx * nz);
  const sizingBodies = bodies?.map((body) => ({ position_m: body.position_m, dimensions_m: body.description.dimensions_m })) ?? scene.rigidBodies;
  // Rigid geometry is a persistent explicit sizing source, as in the paper's
  // examples. Surface features need no blanket refinement: the dynamic sizing
  // evaluates its curvature/velocity demand over each candidate leaf's whole
  // footprint, so a flat surface genuinely coarsens (the paper's headline
  // deep-water case) while edges, blobs, and droplets always register.
  const inflow = scene.fluid.inflow;
  const outlet = inflow ? inflowOutletCenter(inflow) : undefined;
  const terrainHeights = sceneHasTerrain(scene) ? terrainColumnHeights(scene, nx, nz) : undefined;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const worldX = -scene.container.width_m / 2 + (x + 0.5) * h.x, worldZ = -scene.container.depth_m / 2 + (z + 0.5) * h.z;
    for (const body of sizingBodies) {
      const radius = Math.max(body.dimensions_m.x, body.dimensions_m.y, body.dimensions_m.z);
      if (Math.hypot(worldX - body.position_m.x, worldZ - body.position_m.z) <= radius + 2 * Math.max(h.x, h.z)) sizing[x + nx * z] = Math.max(sizing[x + nx * z], 2 / Math.min(h.x, h.z));
    }
    // The nozzle aperture must be resolved before any fluid exists there, or
    // the surface-driven sizing never refines around the emerging jet.
    if (inflow && outlet && Math.hypot(worldX - outlet.x, worldZ - outlet.z) <= inflow.radius_m + 2 * Math.max(h.x, h.z)) {
      sizing[x + nx * z] = Math.max(sizing[x + nx * z], 2 / Math.min(h.x, h.z));
    }
    // Sloping ground (a pool rim, a bank) is a persistent geometric feature
    // like a rigid body: keep the columns crossing it at the finest size so a
    // coarse leaf never straddles a step in the terrain floor.
    if (terrainHeights) {
      const height = terrainHeights[x + nx * z];
      const step = Math.max(
        Math.abs(height - terrainHeights[Math.max(0, x - 1) + nx * z]),
        Math.abs(height - terrainHeights[Math.min(nx - 1, x + 1) + nx * z]),
        Math.abs(height - terrainHeights[x + nx * Math.max(0, z - 1)]),
        Math.abs(height - terrainHeights[x + nx * Math.min(nz - 1, z + 1)])
      );
      if (step > h.y) sizing[x + nx * z] = Math.max(sizing[x + nx * z], 2 / Math.min(h.x, h.z));
    }
  }
  return sizing;
}

function interpolation(samples: TallPressureSample[], y: number) {
  let lower = samples[0], upper = samples[samples.length - 1];
  for (const sample of samples) {
    if (sample.y <= y) lower = sample;
    if (sample.y >= y) { upper = sample; break; }
  }
  if (lower.id === upper.id) return [lower.id, lower.id, 1, 0] as const;
  const weight = (y - lower.y) / Math.max(1, upper.y - lower.y);
  return [lower.id, upper.id, 1 - weight, weight] as const;
}

function bufferWithData(device: GPUDevice, label: string, data: ArrayBufferView, usage = GPUBufferUsage.STORAGE, minimumSize = 4) {
  const size = Math.max(minimumSize, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
  if (data.byteLength) device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

export const quadtreeTallCellProjectionShader = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f, counts: vec4u, solve: vec4f, coupling: vec4u, couplingCounts: vec4u, precondition: vec4u, cellGravity: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowVelocityLength: vec4f, inflowTiming: vec4f }
struct Face { nodes: vec4u, coefficients: vec4f, bounds: vec4u, packed: u32, solidFlux: f32, weights: vec2f, sampleCells: vec4u, sampleSpans: vec4u, flux: f32, mlsMean: f32, volume: f32 }
struct Entry { face: u32, coefficient: f32 }
alias SolverField = u32;
const PRESSURE: SolverField = 0u; const BEST_PRESSURE: SolverField = 1u;
const RESIDUAL: SolverField = 2u; const DIRECTION: SolverField = 3u;
const PRECONDITIONED: SolverField = 4u; const MATRIX_DIRECTION: SolverField = 5u;
const DIAGONAL: SolverField = 6u; const ACTIVE_FLAG: SolverField = 7u;
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var<storage, read_write> faces: array<Face>;
@group(0) @binding(4) var<storage, read> rowOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> rowEntries: array<Entry>;
@group(0) @binding(6) var<storage, read_write> matrixWords: array<u32>;
@group(0) @binding(7) var cellProjection: texture_3d<f32>;
@group(0) @binding(8) var<storage, read_write> state: array<u32>;
@group(0) @binding(9) var<storage, read_write> scalars: array<f32>;
@group(0) @binding(10) var<storage, read_write> factorColumns: array<vec2u>;
@group(0) @binding(11) var<storage, read> factorEntries: array<Entry>;
@group(0) @binding(12) var<uniform> params: Params;
@group(0) @binding(13) var factorAux: texture_2d<u32>;
@group(0) @binding(14) var cellPressureSamples: texture_3d<u32>;
@group(0) @binding(15) var levelSetIn: texture_3d<f32>;
@group(0) @binding(16) var mappedPressureOut: texture_storage_3d<r32float, write>;
@group(0) @binding(17) var mappedPressureIn: texture_3d<f32>;
var<workgroup> reductionA: array<f32, 256>;
var<workgroup> reductionB: array<f32, 256>;
fn inflowGridDims() -> vec3i { return vec3i(params.dims.xyz); }
${inflowBoundaryWGSL}
fn dofs() -> u32 { return params.counts.x; }
fn stateIndex(row: u32, field: SolverField) -> u32 { return field * dofs() + row; }
fn stateF(row: u32, field: SolverField) -> f32 { return bitcast<f32>(state[stateIndex(row, field)]); }
fn setStateF(row: u32, field: SolverField, value: f32) { state[stateIndex(row, field)] = bitcast<u32>(value); }
fn addStateF(row: u32, field: SolverField, value: f32) { setStateF(row, field, stateF(row, field) + value); }
fn matrixEntryBase(entry: u32) -> u32 { return dofs() + 1u + 4u * entry; }
fn matrixStart(row: u32) -> u32 { return matrixWords[row]; }
fn matrixNode(entry: u32) -> u32 { return matrixWords[matrixEntryBase(entry)]; }
fn matrixFaceSlot(entry: u32) -> u32 { return matrixWords[matrixEntryBase(entry) + 1u]; }
fn matrixCoefficient(entry: u32) -> f32 { return bitcast<f32>(matrixWords[matrixEntryBase(entry) + 2u]); }
fn matrixBaseCoefficient(entry: u32) -> f32 { return bitcast<f32>(matrixWords[matrixEntryBase(entry) + 3u]); }
fn setMatrixCoefficient(entry: u32, value: f32) { matrixWords[matrixEntryBase(entry) + 2u] = bitcast<u32>(value); }
fn auxWord(index: u32) -> u32 {
  let texel = index / 4u; let packed = textureLoad(factorAux, vec2i(i32(texel % params.dims.w), i32(texel / params.dims.w)), 0);
  return packed[index % 4u];
}
fn auxEntry(wordOffset: u32, index: u32) -> Entry { return Entry(auxWord(wordOffset + 2u * index), bitcast<f32>(auxWord(wordOffset + 2u * index + 1u))); }
fn faceSpan(face: Face) -> u32 { return face.packed & 0xffffu; }
fn faceAxis(face: Face) -> u32 { return (face.packed >> 16u) & 0x3u; }
fn faceNodeCount(face: Face) -> u32 { return (face.packed >> 18u) & 0x7u; }
fn faceGhost(face: Face) -> bool { return ((face.packed >> 21u) & 0x1u) != 0u; }
fn faceSlotLiquid(face: Face, slot: u32) -> bool { return ((face.packed >> (22u + slot)) & 0x1u) != 0u; }
fn packedSamplePhi(packed: u32, span: u32) -> f32 {
  let origin = vec2u(packed & 1023u, (packed >> 10u) & 1023u); let y = (packed >> 20u) & 1023u;
  let position = vec2f(origin) + vec2f(f32(span) * 0.5 - 0.5);
  let a = vec2u(floor(position)); let b = min(a + vec2u(1), params.dims.xz - vec2u(1)); let t = fract(position);
  let p00 = textureLoad(levelSetIn, vec3i(i32(a.x), i32(y), i32(a.y)), 0).x;
  let p10 = textureLoad(levelSetIn, vec3i(i32(b.x), i32(y), i32(a.y)), 0).x;
  let p01 = textureLoad(levelSetIn, vec3i(i32(a.x), i32(y), i32(b.y)), 0).x;
  let p11 = textureLoad(levelSetIn, vec3i(i32(b.x), i32(y), i32(b.y)), 0).x;
  let centre = mix(mix(p00, p10, t.x), mix(p01, p11, t.x), t.y);
  var footprintMinimum = centre;
  for (var z = origin.y; z < min(origin.y + span, params.dims.z); z += 1u) {
    for (var x = origin.x; x < min(origin.x + span, params.dims.x); x += 1u) {
      let value = textureLoad(levelSetIn, vec3i(i32(x), i32(y), i32(z)), 0).x;
      footprintMinimum = min(footprintMinimum, value);
    }
  }
  return footprintMinimum;
}
fn faceSamplePhi(face: Face, slot: u32) -> f32 {
  return packedSamplePhi(face.sampleCells[slot], face.sampleSpans[slot]);
}
@compute @workgroup_size(128)
fn refreshFaces(@builtin(global_invocation_id) gid: vec3u) {
  let faceId = gid.x; if (faceId >= params.counts.y) { return; }
  var face = faces[faceId]; var all = 0.0; var liquid = 0.0; var allLiquid = true; var liquidMask = 0u;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let phi = faceSamplePhi(face, slot); let term = face.coefficients[slot] * phi; all += term;
    if (phi < 0.0) { liquid += term; liquidMask |= 1u << slot; } else { allLiquid = false; }
  }
  var scale = 1.0;
  if (!allLiquid) { scale = select(min(100.0, max(0.0, all / liquid)), 0.0, abs(liquid) < 1e-12); }
  face.weights.y = face.weights.x * scale;
  face.packed = (face.packed & 0x003fffffu) | (liquidMask << 22u);
  face.flux = face.weights.x * faceVelocity(face) + face.solidFlux;
  faces[faceId] = face;
}
fn faceVelocity(face: Face) -> f32 {
  let axis = faceAxis(face);
  // Every vertical face spans its leaf's full x/z footprint, ghost or not;
  // the horizontal branch below would sample only the corner column.
  if (axis != 1u) {
    var sum = 0.0; var count = 0.0;
    for (var y = face.bounds.z; y < face.bounds.w; y += 1u) {
      for (var transverse = 0u; transverse < faceSpan(face); transverse += 1u) {
        var left = vec3u(face.bounds.x, y, face.bounds.y);
        if (axis == 0u) { left.z += transverse; }
        if (axis == 2u) { left.x += transverse; }
        sum += textureLoad(velocityIn, vec3i(left), 0)[axis]; count += 1.0;
      }
    }
    return sum / max(1.0, count);
  }
  var sum = 0.0; var count = 0.0;
  for (var z = face.bounds.y; z < face.bounds.y + faceSpan(face); z += 1u) {
    for (var x = face.bounds.x; x < face.bounds.x + faceSpan(face); x += 1u) {
      for (var y = face.bounds.z; y < face.bounds.w; y += 1u) {
        sum += textureLoad(velocityIn, vec3i(vec3u(x, y, z)), 0).y; count += 1.0;
      }
    }
  }
  return sum / max(1.0, count);
}
fn faceGradient(face: Face) -> f32 {
  var result = 0.0;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let node = face.nodes[slot]; if (node != 0xffffffffu && faceSlotLiquid(face, slot)) { result += face.coefficients[slot] * stateF(node, DIRECTION); }
  }
  return result;
}
@compute @workgroup_size(128)
fn refreshRows(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; } var rowIsActive = false;
  var openVolume = 0.0; var faceVolume = 0.0;
  for (var entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; entry += 1u) {
    let face = faces[rowEntries[entry].face];
    openVolume += face.weights.x; faceVolume += face.volume;
    for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
      if (face.nodes[slot] == row && faceSlotLiquid(face, slot)) { rowIsActive = true; }
    }
  }
  // Chentanez & Müller Sec. 3.9.1 treats a cell swallowed by a rigid body as
  // solid. A row whose faces are (nearly) all closed by solid fractions has
  // ~zero conductance yet keeps a finite (1-A) u_solid constraint flux, and
  // solving that against the epsilon diagonal produced astronomic pressures
  // around moving bodies. Drop such rows from the solve entirely.
  if (openVolume <= 0.1 * faceVolume) { rowIsActive = false; }
  state[stateIndex(row, ACTIVE_FLAG)] = select(0u, 1u, rowIsActive);
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) {
    let packed = matrixFaceSlot(entry); let face = faces[packed & 0x3fffffffu]; let slot = packed >> 30u;
    let coefficient = select(0.0, matrixBaseCoefficient(entry) * face.weights.y, rowIsActive && faceSlotLiquid(face, slot));
    setMatrixCoefficient(entry, coefficient);
  }
}
fn dofActive(row: u32) -> bool { return state[stateIndex(row, ACTIVE_FLAG)] != 0u; }
fn rowProduct(row: u32) -> f32 {
  if (!dofActive(row)) { return stateF(row, DIRECTION); }
  var sum = 0.0;
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { sum += matrixCoefficient(entry) * stateF(matrixNode(entry), DIRECTION); }
  return sum;
}
// couplingCounts.z packs the warm-start flag in bit 0 and the persistent
// megakernel's hard iteration cap above it. Seed the iterate from last solve's
// mapped cubical pressure, read back at each sample's defining cell through
// the aux dof-sample table. The texture is zero on the first solve and after
// an async topology swap, which degrades gracefully to the cold start.
fn warmStartActive() -> bool { return (params.couplingCounts.z & 1u) != 0u; }
fn dofSampleCell(row: u32) -> vec3i {
  let base = params.couplingCounts.y + 4u * row;
  let word = auxWord(base); let size = max(1u, auxWord(base + 1u));
  return vec3i(i32((word & 1023u) + size / 2u), i32(word >> 20u), i32(((word >> 10u) & 1023u) + size / 2u));
}
fn warmStartPressure(row: u32) -> f32 {
  if (!dofActive(row)) { return 0.0; }
  return textureLoad(mappedPressureIn, dofSampleCell(row), 0).x;
}
fn initializeRow(row: u32) {
  if (!dofActive(row)) {
    setStateF(row, PRESSURE, 0.0); setStateF(row, BEST_PRESSURE, 0.0); setStateF(row, RESIDUAL, 0.0); setStateF(row, DIAGONAL, 1.0);
    setStateF(row, PRECONDITIONED, 0.0); setStateF(row, DIRECTION, 0.0); setStateF(row, MATRIX_DIRECTION, 0.0); return;
  }
  var rhs = 0.0; var diag = 0.0;
  for (var entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; entry += 1u) {
    let item = rowEntries[entry]; let face = faces[item.face];
    // The face flux is A u_fluid (from the staged texture) plus the
    // CPU-integrated (1-A) u_solid constraint flux of moving rigid bodies.
    rhs += item.coefficient * face.flux;
  }
  var pressure0 = 0.0; var residual = rhs;
  if (warmStartActive()) {
    pressure0 = warmStartPressure(row);
    for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { residual -= matrixCoefficient(entry) * warmStartPressure(matrixNode(entry)); }
  }
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { if (matrixNode(entry) == row) { diag += matrixCoefficient(entry); } }
  setStateF(row, PRESSURE, pressure0); setStateF(row, BEST_PRESSURE, pressure0); setStateF(row, RESIDUAL, residual); setStateF(row, DIAGONAL, max(diag, 1e-12));
  // MATRIX_DIRECTION carries b until the first SpMV so the stop test can stay
  // relative to |b| rather than to the (already reduced) warm residual.
  setStateF(row, PRECONDITIONED, 0.0); setStateF(row, DIRECTION, 0.0); setStateF(row, MATRIX_DIRECTION, rhs);
}
// Fixed-degree Chebyshev semi-iteration over D^-1 A. PRESSURE and
// BEST_PRESSURE are the ping-pong vectors; PRECONDITIONED stores the previous
// polynomial correction, MATRIX_DIRECTION stores rho, and DIRECTION retains
// b for the final residual report. Geometry and coefficients were refreshed
// once before this ladder.
fn initializeChebyshevRow(row: u32) {
  initializeRow(row);
  setStateF(row, DIRECTION, stateF(row, MATRIX_DIRECTION));
  setStateF(row, PRECONDITIONED, 0.0);
  setStateF(row, MATRIX_DIRECTION, 0.0);
}
@compute @workgroup_size(128)
fn initializeChebyshev(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < dofs()) { initializeChebyshevRow(gid.x); }
}
fn pressureProduct(row: u32, source: SolverField) -> f32 {
  if (!dofActive(row)) { return stateF(row, source); }
  var sum = 0.0;
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) {
    sum += matrixCoefficient(entry) * stateF(matrixNode(entry), source);
  }
  return sum;
}
fn chebyshevUpdate(row: u32, source: SolverField, destination: SolverField) {
  if (!dofActive(row)) {
    setStateF(row, destination, 0.0); setStateF(row, PRECONDITIONED, 0.0); setStateF(row, MATRIX_DIRECTION, 0.0); return;
  }
  let residual = (stateF(row, DIRECTION) - pressureProduct(row, source)) / stateF(row, DIAGONAL);
  let lower = 0.01; let upper = 2.2;
  let theta = 0.5 * (upper + lower); let delta = 0.5 * (upper - lower); let sigma = theta / delta;
  let previousSearch = stateF(row, PRECONDITIONED); let previousRho = stateF(row, MATRIX_DIRECTION);
  var rho = 1.0 / sigma;
  var search = residual / theta;
  if (previousRho > 0.0) {
    rho = 1.0 / (2.0 * sigma - previousRho);
    search = rho * previousRho * previousSearch + (2.0 * rho / delta) * residual;
  }
  setStateF(row, destination, stateF(row, source) + search);
  setStateF(row, PRECONDITIONED, search); setStateF(row, MATRIX_DIRECTION, rho);
}
@compute @workgroup_size(128)
fn iterateChebyshevAB(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < dofs()) { chebyshevUpdate(gid.x, PRESSURE, BEST_PRESSURE); }
}
@compute @workgroup_size(128)
fn iterateChebyshevBA(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < dofs()) { chebyshevUpdate(gid.x, BEST_PRESSURE, PRESSURE); }
}
@compute @workgroup_size(128)
fn finishChebyshevFromPressure(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  setStateF(row, BEST_PRESSURE, stateF(row, PRESSURE));
  setStateF(row, RESIDUAL, stateF(row, DIRECTION) - pressureProduct(row, PRESSURE));
}
@compute @workgroup_size(128)
fn finishChebyshevFromBest(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  setStateF(row, RESIDUAL, stateF(row, DIRECTION) - pressureProduct(row, BEST_PRESSURE));
}
@compute @workgroup_size(256)
fn reduceChebyshevResidual(@builtin(local_invocation_id) lid: vec3u) {
  var bb = 0.0; var rr = 0.0;
  for (var row = lid.x; row < dofs(); row += 256u) {
    bb += stateF(row, DIRECTION) * stateF(row, DIRECTION);
    rr += stateF(row, RESIDUAL) * stateF(row, RESIDUAL);
  }
  reducePair(lid.x, bb, rr);
  if (lid.x == 0u) {
    scalars[2] = reductionB[0]; scalars[3] = max(reductionA[0], 1e-30); scalars[7] = reductionB[0];
    scalars[9] = f32(params.counts.z);
  }
}
@compute @workgroup_size(128)
fn initialize(@builtin(global_invocation_id) gid: vec3u) { if (gid.x < dofs()) { initializeRow(gid.x); } }
@compute @workgroup_size(128)
fn initializeJacobiDirection(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  initializeRow(row);
  setStateF(row, PRECONDITIONED, stateF(row, RESIDUAL) / stateF(row, DIAGONAL));
  setStateF(row, DIRECTION, stateF(row, PRECONDITIONED));
}
@compute @workgroup_size(128)
fn initializePolynomialStart(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  initializeRow(row);
  setStateF(row, PRECONDITIONED, 0.5 * stateF(row, RESIDUAL) / stateF(row, DIAGONAL));
}
fn applyPrecondition(lid: u32, solveActive: bool) {
  if (solveActive) { for (var row = lid; row < dofs(); row += 256u) { setStateF(row, PRECONDITIONED, stateF(row, RESIDUAL)); } }
  storageBarrier(); workgroupBarrier();
  // The IC graph is level-scheduled so every row within a level is
  // independent. One workgroup supplies the required global ordering while
  // evaluating up to 256 triangular rows concurrently.
  for (var level = 0u; level < params.counts.w; level += 1u) {
    let levelsOffset = bitcast<u32>(params.solve.y); let rowOffsetsOffset = bitcast<u32>(params.solve.z); let rowEntriesOffset = bitcast<u32>(params.solve.w);
    let range = vec2u(auxWord(levelsOffset + 4u * level), auxWord(levelsOffset + 4u * level + 1u));
    if (solveActive) { for (var slot = range.x + lid; slot < range.y; slot += 256u) {
      let row = auxWord(slot); var value = stateF(row, PRECONDITIONED);
      for (var entry = auxWord(rowOffsetsOffset + row); entry < auxWord(rowOffsetsOffset + row + 1u); entry += 1u) {
        let factor = auxEntry(rowEntriesOffset, entry); value -= factor.coefficient * stateF(factor.face, PRECONDITIONED);
      }
      setStateF(row, PRECONDITIONED, value * bitcast<f32>(factorColumns[row].y));
    } }
    storageBarrier(); workgroupBarrier();
  }
  for (var level = 0u; level < params.counts.w; level += 1u) {
    let levelsOffset = bitcast<u32>(params.solve.y);
    let range = vec2u(auxWord(levelsOffset + 4u * level + 2u), auxWord(levelsOffset + 4u * level + 3u));
    if (solveActive) { for (var slot = range.x + lid; slot < range.y; slot += 256u) {
      let column = auxWord(slot); var value = stateF(column, PRECONDITIONED);
      for (var entry = factorColumns[column].x; entry < factorColumns[column + 1u].x; entry += 1u) {
        let factor = factorEntries[entry]; value -= factor.coefficient * stateF(factor.face, PRECONDITIONED);
      }
      setStateF(column, PRECONDITIONED, value * bitcast<f32>(factorColumns[column].y));
    } }
    storageBarrier(); workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn precondition(@builtin(local_invocation_id) lid: vec3u) {
  applyPrecondition(lid.x, !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3]));
}
// Block-restricted IC(0): the CPU factor drops couplings that cross the
// column-aligned row blocks, so each block's triangular solves are
// self-contained. A block's substitution is dominated by the near-serial
// vertical chain of its columns (measured deep-water blocks are single
// ~200-sample columns whose level schedule averages one row per level), so
// one lane owns one whole block: its dependent loads replace the global
// barrier round per level that made the single-workgroup sweep latency-bound,
// and blocks solve in parallel across lanes.
@compute @workgroup_size(64)
fn preconditionBlockIC(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.precondition.y || (scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3])) { return; }
  let header = params.precondition.x + 2u * gid.x;
  let rowStart = auxWord(header); let rowEnd = auxWord(header + 1u);
  let rowOffsetsOffset = bitcast<u32>(params.solve.z); let rowEntriesOffset = bitcast<u32>(params.solve.w);
  for (var row = rowStart; row < rowEnd; row += 1u) {
    var value = stateF(row, RESIDUAL);
    for (var entry = auxWord(rowOffsetsOffset + row); entry < auxWord(rowOffsetsOffset + row + 1u); entry += 1u) {
      let factor = auxEntry(rowEntriesOffset, entry); value -= factor.coefficient * stateF(factor.face, PRECONDITIONED);
    }
    setStateF(row, PRECONDITIONED, value * bitcast<f32>(factorColumns[row].y));
  }
  for (var slot = rowEnd; slot > rowStart; slot -= 1u) {
    let column = slot - 1u; var value = stateF(column, PRECONDITIONED);
    for (var entry = factorColumns[column].x; entry < factorColumns[column + 1u].x; entry += 1u) {
      let factor = factorEntries[entry]; value -= factor.coefficient * stateF(factor.face, PRECONDITIONED);
    }
    setStateF(column, PRECONDITIONED, value * bitcast<f32>(factorColumns[column].y));
  }
}
@compute @workgroup_size(128)
fn preconditionJacobi(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row < dofs() && !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3])) {
    setStateF(row, PRECONDITIONED, stateF(row, RESIDUAL) / stateF(row, DIAGONAL));
  }
}
fn matrixValue(row: u32, node: u32) -> f32 {
  var value = 0.0;
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { if (matrixNode(entry) == node) { value += matrixCoefficient(entry); } }
  return value;
}
@compute @workgroup_size(128)
fn preconditionLine(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.precondition.z || (scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3])) { return; }
  let first = auxWord(params.precondition.x + gid.x); let end = auxWord(params.precondition.x + gid.x + 1u);
  var previous = 0xffffffffu; var previousC = 0.0; var previousD = 0.0;
  for (var slot = first; slot < end; slot += 1u) {
    let row = auxWord(params.precondition.y + slot); var next = 0xffffffffu; if (slot + 1u < end) { next = auxWord(params.precondition.y + slot + 1u); }
    let a = select(0.0, matrixValue(row, previous), previous != 0xffffffffu);
    let c = select(0.0, matrixValue(row, next), next != 0xffffffffu);
    let denominator = max(1e-12, stateF(row, DIAGONAL) - a * previousC);
    let cPrime = c / denominator; let dPrime = (stateF(row, RESIDUAL) - a * previousD) / denominator;
    setStateF(row, MATRIX_DIRECTION, cPrime); setStateF(row, PRECONDITIONED, dPrime);
    previous = row; previousC = cPrime; previousD = dPrime;
  }
  var nextValue = 0.0;
  for (var reverse = end; reverse > first; reverse -= 1u) {
    let row = auxWord(params.precondition.y + reverse - 1u);
    let value = stateF(row, PRECONDITIONED) - stateF(row, MATRIX_DIRECTION) * nextValue;
    setStateF(row, PRECONDITIONED, value); nextValue = value;
  }
}
@compute @workgroup_size(128)
fn preconditionPolynomialStart(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row < dofs() && !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3])) { setStateF(row, PRECONDITIONED, 0.5 * stateF(row, RESIDUAL) / stateF(row, DIAGONAL)); }
}
@compute @workgroup_size(128)
fn preconditionPolynomialMultiply(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  var value = select(0.0, stateF(row, PRECONDITIONED), !dofActive(row));
  if (dofActive(row)) { for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) { value += matrixCoefficient(entry) * stateF(matrixNode(entry), PRECONDITIONED); } }
  setStateF(row, MATRIX_DIRECTION, value);
}
@compute @workgroup_size(128)
fn preconditionPolynomialUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < dofs()) { addStateF(row, PRECONDITIONED, 0.5 * (stateF(row, RESIDUAL) - stateF(row, MATRIX_DIRECTION)) / stateF(row, DIAGONAL)); }
}
@compute @workgroup_size(128)
fn preconditionPolynomialUpdateDirection(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  addStateF(row, PRECONDITIONED, 0.5 * (stateF(row, RESIDUAL) - stateF(row, MATRIX_DIRECTION)) / stateF(row, DIAGONAL));
  setStateF(row, DIRECTION, stateF(row, PRECONDITIONED));
}
@compute @workgroup_size(128)
fn startDirection(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < dofs()) { setStateF(gid.x, DIRECTION, stateF(gid.x, PRECONDITIONED)); }
}
fn reducePair(local: u32, a: f32, b: f32) {
  reductionA[local] = a; reductionB[local] = b; workgroupBarrier();
  var stride = 128u;
  loop { if (local < stride) { reductionA[local] += reductionA[local + stride]; reductionB[local] += reductionB[local + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u; }
}
// Warm-start companion for reduceInitial: captures |b|^2 from the staged
// MATRIX_DIRECTION copy before the preconditioner setup overwrites it, so the
// stop test stays relative to |b| exactly as in the cold start.
@compute @workgroup_size(256)
fn reduceInitialNorm(@builtin(local_invocation_id) lid: vec3u) {
  var bb = 0.0;
  for (var row = lid.x; row < dofs(); row += 256u) { let b = stateF(row, MATRIX_DIRECTION); bb += b * b; }
  reducePair(lid.x, bb, 0.0); if (lid.x == 0u) { scalars[3] = max(reductionA[0], 1e-30); }
}
@compute @workgroup_size(256)
fn reduceInitial(@builtin(local_invocation_id) lid: vec3u) {
  var rz = 0.0; var rr = 0.0;
  for (var row = lid.x; row < dofs(); row += 256u) { rz += stateF(row, RESIDUAL) * stateF(row, PRECONDITIONED); rr += stateF(row, RESIDUAL) * stateF(row, RESIDUAL); }
  reducePair(lid.x, rz, rr);
  if (lid.x == 0u) {
    scalars[0] = reductionA[0]; scalars[2] = reductionB[0]; scalars[7] = reductionB[0];
    if (!warmStartActive()) { scalars[3] = max(reductionB[0], 1e-30); }
  }
}
@compute @workgroup_size(128)
fn multiply(@builtin(global_invocation_id) gid: vec3u) { if (gid.x < dofs()) { setStateF(gid.x, MATRIX_DIRECTION, rowProduct(gid.x)); } }
@compute @workgroup_size(256)
fn applyStep(@builtin(local_invocation_id) lid: vec3u) {
  var value = 0.0; for (var row = lid.x; row < dofs(); row += 256u) { value += stateF(row, DIRECTION) * stateF(row, MATRIX_DIRECTION); }
  reducePair(lid.x, value, 0.0); if (lid.x == 0u) { scalars[1] = reductionA[0]; scalars[4] = select(scalars[0] / max(reductionA[0], 1e-30), 0.0, scalars[2] <= params.solve.x * scalars[3]); }
  storageBarrier(); workgroupBarrier();
  let alpha = scalars[4];
  for (var row = lid.x; row < dofs(); row += 256u) {
    addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION));
    addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION));
  }
}
@compute @workgroup_size(256)
fn finishIteration(@builtin(local_invocation_id) lid: vec3u) {
  var rz = 0.0; var rr = 0.0; for (var row = lid.x; row < dofs(); row += 256u) { rz += stateF(row, RESIDUAL) * stateF(row, PRECONDITIONED); rr += stateF(row, RESIDUAL) * stateF(row, RESIDUAL); }
  reducePair(lid.x, rz, rr); if (lid.x == 0u) { scalars[5] = reductionA[0]; scalars[2] = reductionB[0]; scalars[7] = min(scalars[7], reductionB[0]); scalars[6] = reductionA[0] / max(abs(scalars[0]), 1e-30) * sign(scalars[0]); }
  storageBarrier(); workgroupBarrier();
  for (var row = lid.x; row < dofs(); row += 256u) {
    if (scalars[2] <= scalars[7]) { setStateF(row, BEST_PRESSURE, stateF(row, PRESSURE)); }
    setStateF(row, DIRECTION, stateF(row, PRECONDITIONED) + scalars[6] * stateF(row, DIRECTION));
  }
  if (lid.x == 0u) { scalars[0] = scalars[5]; }
}
fn reducePartial(local: u32, a: f32, b: f32) {
  reductionA[local] = a; reductionB[local] = b; workgroupBarrier();
  var stride = 64u;
  loop { if (local < stride) { reductionA[local] += reductionA[local + stride]; reductionB[local] += reductionB[local + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u; }
}
fn partialBase() -> u32 { return 108u; }
@compute @workgroup_size(128)
fn applyStepPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var value = 0.0; if (gid.x < dofs()) { value = stateF(gid.x, DIRECTION) * stateF(gid.x, MATRIX_DIRECTION); }
  reducePartial(lid.x, value, 0.0); if (lid.x == 0u) { scalars[partialBase() + wid.x] = reductionA[0]; }
}
@compute @workgroup_size(256)
fn applyStepFinalize(@builtin(local_invocation_id) lid: vec3u) {
  var value = 0.0; for (var part = lid.x; part < params.couplingCounts.w; part += 256u) { value += scalars[partialBase() + part]; }
  reducePair(lid.x, value, 0.0); if (lid.x == 0u) { scalars[1] = reductionA[0]; scalars[4] = scalars[0] / max(reductionA[0], 1e-30); }
}
@compute @workgroup_size(128)
fn applyStepUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < dofs()) { let alpha = scalars[4]; addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION)); addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION)); }
}
fn applyStepUpdateRow(row: u32) {
  let alpha = scalars[4]; addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION)); addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION));
}
@compute @workgroup_size(128)
fn applyStepUpdateJacobi(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  applyStepUpdateRow(row);
  setStateF(row, PRECONDITIONED, stateF(row, RESIDUAL) / stateF(row, DIAGONAL));
}
@compute @workgroup_size(128)
fn applyStepUpdatePolynomialStart(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; }
  applyStepUpdateRow(row);
  setStateF(row, PRECONDITIONED, 0.5 * stateF(row, RESIDUAL) / stateF(row, DIAGONAL));
}
@compute @workgroup_size(128)
fn finishIterationPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var rz = 0.0; var rr = 0.0;
  if (gid.x < dofs()) { rz = stateF(gid.x, RESIDUAL) * stateF(gid.x, PRECONDITIONED); rr = stateF(gid.x, RESIDUAL) * stateF(gid.x, RESIDUAL); }
  reducePartial(lid.x, rz, rr); if (lid.x == 0u) { scalars[partialBase() + wid.x] = reductionA[0]; scalars[partialBase() + params.couplingCounts.w + wid.x] = reductionB[0]; }
}
@compute @workgroup_size(128)
fn preconditionPolynomialUpdateFinishPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let row = gid.x; var rz = 0.0; var rr = 0.0;
  if (row < dofs()) {
    addStateF(row, PRECONDITIONED, 0.5 * (stateF(row, RESIDUAL) - stateF(row, MATRIX_DIRECTION)) / stateF(row, DIAGONAL));
    rz = stateF(row, RESIDUAL) * stateF(row, PRECONDITIONED); rr = stateF(row, RESIDUAL) * stateF(row, RESIDUAL);
  }
  reducePartial(lid.x, rz, rr); if (lid.x == 0u) { scalars[partialBase() + wid.x] = reductionA[0]; scalars[partialBase() + params.couplingCounts.w + wid.x] = reductionB[0]; }
}
// The parallel finalizer's bind group aliases binding 10 to dispatchArgs.
// Its direct one-workgroup dispatch can therefore publish next-iteration
// work without aliasing that buffer as an indirect source in the same scope.
fn controlWord(index: u32) -> u32 { return factorColumns[index / 2u][index % 2u]; }
fn setControlWord(index: u32, value: u32) {
  let pairIndex = index / 2u; var pair = factorColumns[pairIndex]; pair[index % 2u] = value; factorColumns[pairIndex] = pair;
}
fn publishNextDispatches(keepSolving: bool) {
  for (var word = 0u; word < 12u; word += 1u) { setControlWord(word, select(0u, controlWord(12u + word), keepSolving || word % 3u != 0u)); }
}
@compute @workgroup_size(256)
fn finishIterationFinalize(@builtin(local_invocation_id) lid: vec3u) {
  let solveActive = scalars[10] != 0.0;
  var rz = 0.0; var rr = 0.0;
  if (solveActive) { for (var part = lid.x; part < params.couplingCounts.w; part += 256u) { rz += scalars[partialBase() + part]; rr += scalars[partialBase() + params.couplingCounts.w + part]; } }
  reducePair(lid.x, rz, rr);
  if (lid.x == 0u) {
    if (solveActive) {
      scalars[5] = reductionA[0]; scalars[2] = reductionB[0]; scalars[7] = min(scalars[7], reductionB[0]);
      scalars[6] = reductionA[0] / max(abs(scalars[0]), 1e-30) * sign(scalars[0]); scalars[0] = reductionA[0];
      let keepSolving = !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3]);
      if (keepSolving) { scalars[9] += 1.0; }
      scalars[10] = select(0.0, 1.0, keepSolving); publishNextDispatches(keepSolving);
      // This triple belongs to the iteration just finalized, so it remains
      // active even when the next-iteration triples were switched off.
      setControlWord(31u, controlWord(12u)); setControlWord(32u, 1u); setControlWord(33u, 1u);
    } else {
      setControlWord(31u, 0u); setControlWord(32u, 1u); setControlWord(33u, 1u);
    }
  }
}
@compute @workgroup_size(128)
fn finishIterationUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < dofs()) { if (scalars[2] <= scalars[7]) { setStateF(row, BEST_PRESSURE, stateF(row, PRESSURE)); } setStateF(row, DIRECTION, stateF(row, PRECONDITIONED) + scalars[6] * stateF(row, DIRECTION)); }
}
// Fused SpMV + d.Ad partial for the uncoupled parallel path. The coupled path
// keeps the separate applyStepPartial because coupleApply modifies
// [MATRIX_DIRECTION] between the multiply and the dot product.
@compute @workgroup_size(128)
fn multiplyPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  var value = 0.0;
  if (gid.x < dofs()) {
    let product = rowProduct(gid.x);
    setStateF(gid.x, MATRIX_DIRECTION, product);
    value = stateF(gid.x, DIRECTION) * product;
  }
  reducePartial(lid.x, value, 0.0); if (lid.x == 0u) { scalars[partialBase() + wid.x] = reductionA[0]; }
}
// Every workgroup re-reduces the d.Ad partials instead of waiting on a
// dedicated finalize dispatch: the partial list is tiny (one value per
// 128 rows) and stays L2-resident, so the redundant sum is far cheaper than
// another dispatch boundary.
var<workgroup> alphaShared: f32;
fn broadcastAlpha(lid: u32) -> f32 {
  if (lid == 0u) {
    var denominator = 0.0;
    for (var part = 0u; part < params.couplingCounts.w; part += 1u) { denominator += scalars[partialBase() + part]; }
    alphaShared = scalars[0] / max(denominator, 1e-30);
  }
  workgroupBarrier();
  return alphaShared;
}
@compute @workgroup_size(128)
fn applyStepAlphaUpdate(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let alpha = broadcastAlpha(lid.x);
  let row = gid.x; if (row >= dofs()) { return; }
  addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION));
  addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION));
  if (row == 0u) { scalars[4] = alpha; }
}
@compute @workgroup_size(128)
fn applyStepAlphaUpdatePolynomialStart(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let alpha = broadcastAlpha(lid.x);
  let row = gid.x; if (row >= dofs()) { return; }
  addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION));
  addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION));
  setStateF(row, PRECONDITIONED, 0.5 * stateF(row, RESIDUAL) / stateF(row, DIAGONAL));
  if (row == 0u) { scalars[4] = alpha; }
}
@compute @workgroup_size(128)
fn applyStepAlphaUpdateJacobiFinishPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let alpha = broadcastAlpha(lid.x);
  let row = gid.x; var rz = 0.0; var rr = 0.0;
  if (row < dofs()) {
    addStateF(row, PRESSURE, alpha * stateF(row, DIRECTION));
    addStateF(row, RESIDUAL, -alpha * stateF(row, MATRIX_DIRECTION));
    let preconditioned = stateF(row, RESIDUAL) / stateF(row, DIAGONAL);
    setStateF(row, PRECONDITIONED, preconditioned);
    rz = stateF(row, RESIDUAL) * preconditioned; rr = stateF(row, RESIDUAL) * stateF(row, RESIDUAL);
    if (row == 0u) { scalars[4] = alpha; }
  }
  reducePartial(lid.x, rz, rr); if (lid.x == 0u) { scalars[partialBase() + wid.x] = reductionA[0]; scalars[partialBase() + params.couplingCounts.w + wid.x] = reductionB[0]; }
}
// Persistent-CG path for uncoupled polynomial/Jacobi solves. One workgroup
// owns every phase, so storage barriers provide global ordering without a
// dispatch boundary. params.precondition.w is 1 for Jacobi and 2--4 for the
// damped-Jacobi polynomial degree.
fn megakernelPreconditionedProduct(row: u32) -> f32 {
  if (!dofActive(row)) { return stateF(row, PRECONDITIONED); }
  var sum = 0.0;
  for (var entry = matrixStart(row); entry < matrixStart(row + 1u); entry += 1u) {
    sum += matrixCoefficient(entry) * stateF(matrixNode(entry), PRECONDITIONED);
  }
  return sum;
}
var<workgroup> megakernelBb: f32;
var<workgroup> megakernelRz: f32;
var<workgroup> megakernelRr: f32;
var<workgroup> megakernelMinimumRr: f32;
var<workgroup> megakernelAlpha: f32;
var<workgroup> megakernelBeta: f32;
var<workgroup> megakernelConverged: u32;
var<workgroup> megakernelSnapshotBest: u32;
@compute @workgroup_size(256)
fn solveMegakernel(@builtin(local_invocation_id) lid: vec3u) {
  let local = lid.x;
  let degree = params.precondition.w;
  let hardBudget = params.couplingCounts.z >> 1u;

  // Initialize the warm/cold iterate and preserve b in MATRIX_DIRECTION.
  for (var row = local; row < dofs(); row += 256u) { initializeRow(row); }
  storageBarrier(); workgroupBarrier();

  // Capture |b|^2 before polynomial scratch reuses MATRIX_DIRECTION, and
  // capture the initial residual independently for the best-iterate guard.
  var bb = 0.0; var rr0 = 0.0;
  for (var row = local; row < dofs(); row += 256u) {
    let b = stateF(row, MATRIX_DIRECTION); let r = stateF(row, RESIDUAL);
    bb += b * b; rr0 += r * r;
  }
  reducePair(local, bb, rr0);
  if (local == 0u) {
    megakernelBb = max(reductionA[0], 1e-30);
    megakernelRr = reductionB[0]; megakernelMinimumRr = reductionB[0];
    scalars[2] = reductionB[0]; scalars[3] = megakernelBb; scalars[7] = reductionB[0];
    scalars[4] = 0.0; scalars[6] = 0.0; scalars[9] = 0.0;
  }
  storageBarrier(); workgroupBarrier();

  // z0 = M^-1 r0. Degree one is Jacobi; higher degrees evaluate the same
  // damped Neumann polynomial as the dispatch ladder.
  for (var row = local; row < dofs(); row += 256u) {
    let scaled = stateF(row, RESIDUAL) / stateF(row, DIAGONAL);
    setStateF(row, PRECONDITIONED, select(0.5 * scaled, scaled, degree == 1u));
  }
  storageBarrier(); workgroupBarrier();
  for (var polynomialPass = 1u; polynomialPass < degree; polynomialPass += 1u) {
    for (var row = local; row < dofs(); row += 256u) {
      setStateF(row, MATRIX_DIRECTION, megakernelPreconditionedProduct(row));
    }
    storageBarrier(); workgroupBarrier();
    for (var row = local; row < dofs(); row += 256u) {
      addStateF(row, PRECONDITIONED, 0.5 * (stateF(row, RESIDUAL) - stateF(row, MATRIX_DIRECTION)) / stateF(row, DIAGONAL));
    }
    storageBarrier(); workgroupBarrier();
  }

  var rz0 = 0.0;
  for (var row = local; row < dofs(); row += 256u) {
    let z = stateF(row, PRECONDITIONED);
    setStateF(row, DIRECTION, z); rz0 += stateF(row, RESIDUAL) * z;
  }
  reducePair(local, rz0, 0.0);
  if (local == 0u) {
    megakernelRz = reductionA[0]; scalars[0] = reductionA[0];
    megakernelConverged = select(0u, 1u, megakernelRr <= params.solve.x * megakernelBb);
  }
  storageBarrier(); workgroupBarrier();

  for (var iteration = 0u; iteration < hardBudget; iteration += 1u) {
    // The collective load makes the break condition uniform for every
    // invocation, which is mandatory because all later phases contain
    // workgroup and storage barriers.
    if (workgroupUniformLoad(&megakernelConverged) != 0u) { break; }

    // q = A d and d.q in one strided pass.
    var pAp = 0.0;
    for (var row = local; row < dofs(); row += 256u) {
      let product = rowProduct(row); setStateF(row, MATRIX_DIRECTION, product);
      pAp += stateF(row, DIRECTION) * product;
    }
    reducePair(local, pAp, 0.0);
    if (local == 0u) {
      megakernelAlpha = megakernelRz / max(reductionA[0], 1e-30);
      scalars[1] = reductionA[0]; scalars[4] = megakernelAlpha;
    }
    storageBarrier(); workgroupBarrier();

    for (var row = local; row < dofs(); row += 256u) {
      addStateF(row, PRESSURE, megakernelAlpha * stateF(row, DIRECTION));
      addStateF(row, RESIDUAL, -megakernelAlpha * stateF(row, MATRIX_DIRECTION));
    }
    if (local == 0u) { scalars[9] = f32(iteration + 1u); }
    storageBarrier(); workgroupBarrier();

    // z = M^-1 r, reusing MATRIX_DIRECTION as polynomial scratch now that q
    // has already been consumed by the pressure/residual update.
    for (var row = local; row < dofs(); row += 256u) {
      let scaled = stateF(row, RESIDUAL) / stateF(row, DIAGONAL);
      setStateF(row, PRECONDITIONED, select(0.5 * scaled, scaled, degree == 1u));
    }
    storageBarrier(); workgroupBarrier();
    for (var polynomialPass = 1u; polynomialPass < degree; polynomialPass += 1u) {
      for (var row = local; row < dofs(); row += 256u) {
        setStateF(row, MATRIX_DIRECTION, megakernelPreconditionedProduct(row));
      }
      storageBarrier(); workgroupBarrier();
      for (var row = local; row < dofs(); row += 256u) {
        addStateF(row, PRECONDITIONED, 0.5 * (stateF(row, RESIDUAL) - stateF(row, MATRIX_DIRECTION)) / stateF(row, DIAGONAL));
      }
      storageBarrier(); workgroupBarrier();
    }

    var rz = 0.0; var rr = 0.0;
    for (var row = local; row < dofs(); row += 256u) {
      let r = stateF(row, RESIDUAL);
      rz += r * stateF(row, PRECONDITIONED); rr += r * r;
    }
    reducePair(local, rz, rr);
    if (local == 0u) {
      let nextRz = reductionA[0]; let nextRr = reductionB[0];
      megakernelBeta = nextRz / max(abs(megakernelRz), 1e-30) * sign(megakernelRz);
      megakernelSnapshotBest = select(0u, 1u, nextRr <= megakernelMinimumRr);
      megakernelRz = nextRz; megakernelRr = nextRr; megakernelMinimumRr = min(megakernelMinimumRr, nextRr);
      scalars[0] = nextRz; scalars[2] = nextRr; scalars[5] = nextRz;
      scalars[6] = megakernelBeta; scalars[7] = megakernelMinimumRr;
      megakernelConverged = select(0u, 1u, nextRr <= params.solve.x * megakernelBb);
    }
    storageBarrier(); workgroupBarrier();

    for (var row = local; row < dofs(); row += 256u) {
      if (megakernelSnapshotBest != 0u) { setStateF(row, BEST_PRESSURE, stateF(row, PRESSURE)); }
      setStateF(row, DIRECTION, stateF(row, PRECONDITIONED) + megakernelBeta * stateF(row, DIRECTION));
    }
    storageBarrier(); workgroupBarrier();
  }
}
fn cellIndex(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
// Narita Sec. 4.4: the monolithic body coupling K = [grad]^T [V] (1-[A]) [L] is
// rank six per body. Static rows live in the aux words texture; the per-body
// six-vectors live past the CG scalars (base word 12, stride 8 per body).
fn auxF32(index: u32) -> f32 { return bitcast<f32>(auxWord(index)); }
var<workgroup> coupleScratch: array<f32, 256>;
fn coupleGather(lid: u32, body: u32, usePressure: bool) {
  let offsets = params.coupling.x;
  let start = auxWord(offsets + body); let end = auxWord(offsets + body + 1u);
  let entries = offsets + params.coupling.w + 1u;
  var sums = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var slot = start + lid; slot < end; slot += 256u) {
    let base = entries + slot * 8u; let dof = auxWord(base);
    var value = select(stateF(dof, DIRECTION), stateF(dof, BEST_PRESSURE), usePressure);
    if (!dofActive(dof)) { value = 0.0; }
    for (var component = 0u; component < 6u; component += 1u) { sums[component] += auxF32(base + 2u + component) * value; }
  }
  for (var component = 0u; component < 6u; component += 1u) {
    coupleScratch[lid] = sums[component]; workgroupBarrier();
    var stride = 128u;
    loop { if (lid < stride) { coupleScratch[lid] += coupleScratch[lid + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u; }
    if (lid == 0u) { scalars[12u + body * 8u + component] = coupleScratch[0]; }
    workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn coupleReduce(@builtin(local_invocation_id) lid: vec3u) {
  for (var body = 0u; body < params.coupling.w; body += 1u) { coupleGather(lid.x, body, false); }
  storageBarrier(); workgroupBarrier();
  if (lid.x < params.coupling.w) {
    let body = lid.x; let table = params.coupling.z + body * 12u; let base = 12u + body * 8u;
    let g3 = scalars[base + 3u]; let g4 = scalars[base + 4u]; let g5 = scalars[base + 5u];
    scalars[base] *= auxF32(table); scalars[base + 1u] *= auxF32(table); scalars[base + 2u] *= auxF32(table);
    scalars[base + 3u] = auxF32(table + 1u) * g3 + auxF32(table + 2u) * g4 + auxF32(table + 3u) * g5;
    scalars[base + 4u] = auxF32(table + 4u) * g3 + auxF32(table + 5u) * g4 + auxF32(table + 6u) * g5;
    scalars[base + 5u] = auxF32(table + 7u) * g3 + auxF32(table + 8u) * g4 + auxF32(table + 9u) * g5;
  }
}
@compute @workgroup_size(128)
fn coupleApply(@builtin(global_invocation_id) gid: vec3u) {
  let distinct = params.couplingCounts.x;
  if (gid.x >= distinct) { return; }
  let dofIds = params.coupling.y; let starts = dofIds + distinct; let entries = starts + distinct + 1u;
  let dof = auxWord(dofIds + gid.x);
  var sum = 0.0;
  for (var slot = auxWord(starts + gid.x); slot < auxWord(starts + gid.x + 1u); slot += 1u) {
    let base = entries + slot * 8u; let body = auxWord(base);
    for (var component = 0u; component < 6u; component += 1u) { sum += auxF32(base + 2u + component) * scalars[12u + body * 8u + component]; }
  }
  if (dofActive(dof)) { addStateF(dof, MATRIX_DIRECTION, sum); }
}
@compute @workgroup_size(256)
fn coupleImpulse(@builtin(local_invocation_id) lid: vec3u) {
  // Raw K^T p per body; the host converts to impulses via -rho and M^-1.
  for (var body = 0u; body < params.coupling.w; body += 1u) { coupleGather(lid.x, body, true); }
}
fn solvedFaceGradient(face: Face) -> f32 {
  var result = 0.0;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let node = face.nodes[slot]; if (node != 0xffffffffu && faceSlotLiquid(face, slot)) { result += face.coefficients[slot] * stateF(node, BEST_PRESSURE); }
  }
  return result;
}
fn sampleBasisAndSize(dof: u32, cell: vec3u) -> array<vec4f, 2> {
  let base = params.couplingCounts.y + 4u * dof;
  let packed = auxWord(base); let span = max(1u, auxWord(base + 1u)); let ySpan = max(1u, auxWord(base + 2u));
  let origin = vec2u(packed & 1023u, (packed >> 10u) & 1023u); let sampleY = (packed >> 20u) & 1023u;
  let delta = vec3f(
    f32(origin.x) + 0.5 * f32(span) - (f32(cell.x) + 0.5),
    f32(sampleY) - f32(cell.y),
    f32(origin.y) + 0.5 * f32(span) - (f32(cell.z) + 0.5));
  return array<vec4f, 2>(vec4f(delta, 1.0), vec4f(f32(span), f32(ySpan), f32(span), 0.0));
}
fn representedBasisAndSize(word: u32, packedY: u32, cell: vec3u) -> array<vec4f, 2> {
  let origin = vec2u(word & 1023u, (word >> 10u) & 1023u); let span = max(1u, word >> 20u);
  let first = packedY & 1023u; let last = (packedY >> 10u) & 1023u; let sampleY = select(first, last, cell.y - first > last - cell.y);
  let delta = vec3f(
    f32(origin.x) + 0.5 * f32(span) - (f32(cell.x) + 0.5),
    f32(sampleY) - f32(cell.y),
    f32(origin.y) + 0.5 * f32(span) - (f32(cell.z) + 0.5));
  return array<vec4f, 2>(vec4f(delta, 1.0), vec4f(f32(span), f32(max(1u, last - first + 1u)), f32(span), f32(sampleY)));
}
fn dofSamplePhi(dof: u32) -> f32 {
  let base = params.couplingCounts.y + 4u * dof;
  return packedSamplePhi(auxWord(base), max(1u, auxWord(base + 1u)));
}
fn mlsKernel(basis: vec4f, size: vec4f) -> f32 {
  return max(1.0 - abs(basis.x) / size.x, 0.01)
    * max(1.0 - abs(basis.y) / size.y, 0.01)
    * max(1.0 - abs(basis.z) / size.z, 0.01);
}
fn solveMls4(matrixIn: array<vec4f, 4>) -> vec4f {
  var a = matrixIn; var b = vec4f(0.0, 0.0, 0.0, 1.0);
  for (var pivot = 0u; pivot < 4u; pivot += 1u) {
    var best = pivot;
    for (var row = pivot + 1u; row < 4u; row += 1u) { if (abs(a[row][pivot]) > abs(a[best][pivot])) { best = row; } }
    if (abs(a[best][pivot]) < 1e-8) { return vec4f(3.402823e38); }
    if (best != pivot) { let swap = a[pivot]; a[pivot] = a[best]; a[best] = swap; let sb = b[pivot]; b[pivot] = b[best]; b[best] = sb; }
    let inverse = 1.0 / a[pivot][pivot]; a[pivot] *= inverse; b[pivot] *= inverse;
    for (var row = 0u; row < 4u; row += 1u) {
      if (row == pivot) { continue; }
      let factor = a[row][pivot]; a[row] -= factor * a[pivot]; b[row] -= factor * b[pivot];
    }
  }
  return b;
}
@compute @workgroup_size(4,4,4)
fn mapPressure(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let own = textureLoad(cellPressureSamples, vec3i(gid), 0);
  let ownSpan = max(1u, own.w >> 20u); var ids: array<u32, 64>; var count = 0u;
  var ghostWords: array<u32, 64>; var ghostPackedY: array<u32, 64>; var ghostCount = 0u;
  for (var dz = -1; dz <= 1; dz += 1) { for (var dx = -1; dx <= 1; dx += 1) { for (var dy = -2; dy <= 2; dy += 1) {
    let q = clamp(vec3i(gid) + vec3i(dx * i32(2u * ownSpan), dy, dz * i32(2u * ownSpan)), vec3i(0), vec3i(params.dims.xyz) - vec3i(1));
    let endpoints = textureLoad(cellPressureSamples, q, 0);
    for (var endpoint = 0u; endpoint < 2u; endpoint += 1u) {
      let dof = endpoints[endpoint];
      if (dof != 0xffffffffu && dofActive(dof)) {
        var duplicate = false; for (var slot = 0u; slot < count; slot += 1u) { if (ids[slot] == dof) { duplicate = true; } }
        if (!duplicate && count < 64u) { ids[count] = dof; count += 1u; }
      } else {
        let geometry = representedBasisAndSize(endpoints.w, endpoints.z, gid); let sampleY = u32(geometry[1].w);
        let packed = (endpoints.w & 0x000fffffu) | (sampleY << 20u); let phiAir = packedSamplePhi(packed, max(1u, endpoints.w >> 20u));
        if (phiAir >= 0.0 && phiAir <= 2.0 * min(params.cell.x, min(params.cell.y, params.cell.z))) {
          var duplicateGhost = false; for (var slot = 0u; slot < ghostCount; slot += 1u) { if (ghostWords[slot] == endpoints.w && ghostPackedY[slot] == endpoints.z) { duplicateGhost = true; } }
          if (!duplicateGhost && ghostCount < 64u) { ghostWords[ghostCount] = endpoints.w; ghostPackedY[ghostCount] = endpoints.z; ghostCount += 1u; }
        }
      }
    }
  } } }
  var matrix: array<vec4f, 4>; var totalWeight = 0.0; var shepard = 0.0;
  for (var slot = 0u; slot < count; slot += 1u) {
    let geometry = sampleBasisAndSize(ids[slot], gid); let basis = geometry[0]; let weight = mlsKernel(basis, geometry[1]);
    matrix[0] += weight * basis.x * basis; matrix[1] += weight * basis.y * basis;
    matrix[2] += weight * basis.z * basis; matrix[3] += weight * basis.w * basis;
    totalWeight += weight; shepard += weight * stateF(ids[slot], BEST_PRESSURE);
  }
  var ghostValues: array<f32, 64>;
  for (var ghost = 0u; ghost < ghostCount; ghost += 1u) {
    let geometry = representedBasisAndSize(ghostWords[ghost], ghostPackedY[ghost], gid); let basis = geometry[0]; let weight = mlsKernel(basis, geometry[1]);
    var nearest = 0xffffffffu; var nearestDistance = 3.402823e38;
    for (var slot = 0u; slot < count; slot += 1u) {
      let liquidGeometry = sampleBasisAndSize(ids[slot], gid); let distance = dot(liquidGeometry[0].xyz - basis.xyz, liquidGeometry[0].xyz - basis.xyz);
      let phiLiquid = dofSamplePhi(ids[slot]); if (phiLiquid < 0.0 && distance < nearestDistance) { nearest = ids[slot]; nearestDistance = distance; }
    }
    if (nearest == 0xffffffffu || nearestDistance > 4.01) { continue; }
    let sampleY = u32(geometry[1].w); let packed = (ghostWords[ghost] & 0x000fffffu) | (sampleY << 20u);
    let phiAir = packedSamplePhi(packed, max(1u, ghostWords[ghost] >> 20u)); let phiLiquid = min(dofSamplePhi(nearest), -1e-6);
    // Linear ghost-fluid extrapolation places p=0 at phi=0. The ratio is
    // negative because the resident level set is negative in liquid.
    let ghostValue = clamp(phiAir / phiLiquid, -${maximumVelocityUpdateFluidScale.toFixed(1)}, 0.0) * stateF(nearest, BEST_PRESSURE);
    ghostValues[ghost] = ghostValue;
    matrix[0] += weight * basis.x * basis; matrix[1] += weight * basis.y * basis;
    matrix[2] += weight * basis.z * basis; matrix[3] += weight * basis.w * basis;
    totalWeight += weight; shepard += weight * ghostValue;
  }
  let solved = solveMls4(matrix); var mapped = 0.0;
  if (solved.x > 1e30) { mapped = shepard / max(totalWeight, 1e-12); }
  else {
    for (var slot = 0u; slot < count; slot += 1u) { let geometry = sampleBasisAndSize(ids[slot], gid); mapped += mlsKernel(geometry[0], geometry[1]) * dot(geometry[0], solved) * stateF(ids[slot], BEST_PRESSURE); }
    for (var ghost = 0u; ghost < ghostCount; ghost += 1u) { let geometry = representedBasisAndSize(ghostWords[ghost], ghostPackedY[ghost], gid); mapped += mlsKernel(geometry[0], geometry[1]) * dot(geometry[0], solved) * ghostValues[ghost]; }
  }
  textureStore(mappedPressureOut, vec3i(gid), vec4f(mapped));
}
fn mappedGradient(cell: vec3u, axis: u32) -> f32 {
  var plus = cell; plus[axis] += 1u;
  return (textureLoad(mappedPressureIn, vec3i(plus), 0).x - textureLoad(mappedPressureIn, vec3i(cell), 0).x) / params.cell[axis];
}
fn faceSubfaceCount(face: Face) -> u32 {
  let ySpan = face.bounds.w - face.bounds.z;
  return select(faceSpan(face) * ySpan, faceSpan(face) * faceSpan(face) * ySpan, faceAxis(face) == 1u);
}
@compute @workgroup_size(128)
fn refreshFaceMls(@builtin(global_invocation_id) gid: vec3u) {
  let faceId = gid.x; if (faceId >= params.counts.y) { return; }
  var face = faces[faceId]; let axis = faceAxis(face); var sum = 0.0; var count = 0.0;
  if (axis == 1u) {
    for (var z = face.bounds.y; z < face.bounds.y + faceSpan(face); z += 1u) { for (var x = face.bounds.x; x < face.bounds.x + faceSpan(face); x += 1u) { for (var y = face.bounds.z; y < face.bounds.w; y += 1u) {
      sum += mappedGradient(vec3u(x, y, z), axis); count += 1.0;
    } } }
  } else {
    for (var y = face.bounds.z; y < face.bounds.w; y += 1u) { for (var transverse = 0u; transverse < faceSpan(face); transverse += 1u) {
      var cell = vec3u(face.bounds.x, y, face.bounds.y); if (axis == 0u) { cell.z += transverse; } else { cell.x += transverse; }
      sum += mappedGradient(cell, axis); count += 1.0;
    } }
  }
  face.mlsMean = sum / max(1.0, count); faces[faceId] = face;
}
@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; } let id = vec3i(gid);
  var value = textureLoad(velocityIn, id, 0).xyz; let projection = textureLoad(cellProjection, id, 0);
  let ownPhi = textureLoad(levelSetIn, id, 0).x; let ownLiquid = ownPhi < 0.0;
  let h = min(params.cell.x, min(params.cell.y, params.cell.z));
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var plus = gid; plus[axis] += 1u;
    if (plus[axis] >= params.dims[axis]) { value[axis] = 0.0; continue; }
    let otherPhi = textureLoad(levelSetIn, vec3i(plus), 0).x; let otherLiquid = otherPhi < 0.0;
    // Air immediately outside the interface is the transport field used by
    // the level-set backtrace. Only true far-field air is zeroed here; the
    // five-ring pass following projection fills this narrow band from
    // liquid-touching faces.
    if (!ownLiquid && !otherLiquid && ownPhi > 5.0 * h && otherPhi > 5.0 * h) { value[axis] = 0.0; continue; }
    let packedFace = u32(round(projection[axis])); var gradient = mappedGradient(gid, axis);
    if (packedFace > 0u) {
      let face = faces[packedFace - 1u];
      let fluidScale = min(${maximumVelocityUpdateFluidScale.toFixed(1)}, select(0.0, face.weights.y / face.weights.x, face.weights.x > 0.0));
      let solved = solvedFaceGradient(face);
      if (faceSubfaceCount(face) > 1u) {
        // refreshFaces integrated this same fine velocity field, so its
        // sub-face deviations from the mean sum to zero. Applying only the
        // MLS-varying solved gradient preserves the coarse constrained flux
        // exactly without box-filtering away vertical/horizontal shear.
        gradient = gradient - face.mlsMean + solved;
      } else { gradient = solved; }
      value[axis] -= fluidScale * gradient;
    } else {
      let ownCell = textureLoad(cellPressureSamples, id, 0); let otherCell = textureLoad(cellPressureSamples, vec3i(plus), 0);
      let sameAdaptiveCell = ownCell.z == otherCell.z && ownCell.w == otherCell.w;
      if (sameAdaptiveCell && ownLiquid && otherLiquid) { value[axis] -= gradient; }
    }
  }
  // The nozzle is a prescribed boundary: re-impose its velocity after the
  // pressure gradient (and after the air-air zeroing above), mirroring the
  // restricted method's project kernel.
  value = applyInflowVelocity(id, value);
  textureStore(velocityOut, id, vec4f(value, 0.0));
}
`;

// Convergence-driven indirect dispatch arguments live in their own buffer and
// pipeline. That separation is intentional: WebGPU validates each dispatch as
// one usage scope, so the same buffer cannot be a writable storage binding and
// the indirect argument source of that dispatch. A tiny preceding dispatch
// writes the arguments; all pressure kernels can then remain in one pass.
/**
 * Geometric multigrid is a separate module so the already-full projection
 * layout does not acquire more storage bindings. Coarse matrices are numeric
 * Galerkin products of the refreshed fine operator; the symbolic entry map is
 * rebuilt only when quadtree topology changes.
 */
export const quadtreeMultigridShader = /* wgsl */ `
struct MGParams {
  source: vec4u,
  coarse: vec4u,
  topologyOffsets: vec4u,
  counts: vec4u,
  transferOffsets: vec4u,
  solve: vec4f,
}
@group(0) @binding(0) var<storage, read_write> sourceMatrix: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> targetMatrix: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> sourceVectors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> targetVectors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> topology: array<u32>;
@group(0) @binding(5) var<storage, read> scalars: array<f32>;
@group(0) @binding(7) var<storage, read_write> fineState: array<u32>;
@group(0) @binding(8) var<uniform> params: MGParams;
const RHS = 0u; const CORRECTION = 1u; const DEFECT = 2u;
const CPRIME = 3u; const DPRIME = 4u;
fn sourceCount() -> u32 { return params.source.x; }
fn targetCount() -> u32 { return params.coarse.x; }
fn sourceEntryBase(entry: u32) -> u32 { return sourceCount() + 1u + params.source.z * entry; }
fn targetEntryBase(entry: u32) -> u32 { return targetCount() + 1u + params.coarse.z * entry; }
fn sourceRowStart(row: u32) -> u32 { return atomicLoad(&sourceMatrix[row]); }
fn targetRowStart(row: u32) -> u32 { return atomicLoad(&targetMatrix[row]); }
fn sourceColumn(entry: u32) -> u32 { return atomicLoad(&sourceMatrix[sourceEntryBase(entry)]); }
fn sourceCoefficient(entry: u32) -> f32 { return bitcast<f32>(atomicLoad(&sourceMatrix[sourceEntryBase(entry) + params.source.w])); }
fn targetCoefficient(entry: u32) -> f32 { return bitcast<f32>(atomicLoad(&targetMatrix[targetEntryBase(entry) + params.coarse.w])); }
fn sourceVectorIndex(field: u32, row: u32) -> u32 { return field * sourceCount() + row; }
fn targetVectorIndex(field: u32, row: u32) -> u32 { return field * targetCount() + row; }
fn sourceF(field: u32, row: u32) -> f32 { return bitcast<f32>(atomicLoad(&sourceVectors[sourceVectorIndex(field, row)])); }
fn targetF(field: u32, row: u32) -> f32 { return bitcast<f32>(atomicLoad(&targetVectors[targetVectorIndex(field, row)])); }
fn setSourceF(field: u32, row: u32, value: f32) { atomicStore(&sourceVectors[sourceVectorIndex(field, row)], bitcast<u32>(value)); }
fn setTargetF(field: u32, row: u32, value: f32) { atomicStore(&targetVectors[targetVectorIndex(field, row)], bitcast<u32>(value)); }
fn fineActive(row: u32) -> bool { return params.counts.y == 0u || fineState[7u * sourceCount() + row] != 0u; }
fn solveActive() -> bool { return !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3]); }
fn nodeParent(row: u32) -> u32 { return topology[params.topologyOffsets.x + row]; }
fn entryParent(entry: u32) -> u32 { return topology[params.topologyOffsets.y + entry]; }
fn atomicAddFloat(destination: ptr<storage, atomic<u32>, read_write>, value: f32) {
  if (value == 0.0) { return; }
  var old = atomicLoad(destination);
  loop {
    let next = bitcast<u32>(bitcast<f32>(old) + value);
    let result = atomicCompareExchangeWeak(destination, old, next);
    if (result.exchanged) { break; }
    old = result.old_value;
  }
}
fn sourceMatrixValue(row: u32, column: u32) -> f32 {
  if (!fineActive(row)) { return select(0.0, 1.0, row == column); }
  var value = 0.0;
  for (var entry = sourceRowStart(row); entry < sourceRowStart(row + 1u); entry += 1u) {
    if (sourceColumn(entry) == column) { value += sourceCoefficient(entry); }
  }
  return value;
}
fn sourceLineValues(row: u32, previous: u32, next: u32) -> vec3f {
  if (!fineActive(row)) { return vec3f(0.0, 1.0, 0.0); }
  var result = vec3f(0.0);
  for (var entry = sourceRowStart(row); entry < sourceRowStart(row + 1u); entry += 1u) {
    let column = sourceColumn(entry); let coefficient = sourceCoefficient(entry);
    if (column == previous) { result.x += coefficient; }
    if (column == row) { result.y += coefficient; }
    if (column == next) { result.z += coefficient; }
  }
  return result;
}
fn sourceProduct(row: u32) -> f32 {
  if (!fineActive(row)) { return sourceF(CORRECTION, row); }
  var value = 0.0;
  for (var entry = sourceRowStart(row); entry < sourceRowStart(row + 1u); entry += 1u) {
    value += sourceCoefficient(entry) * sourceF(CORRECTION, sourceColumn(entry));
  }
  return value;
}

@compute @workgroup_size(128)
fn clearCoarseMatrix(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= targetCount()) { return; }
  for (var entry = targetRowStart(row); entry < targetRowStart(row + 1u); entry += 1u) {
    atomicStore(&targetMatrix[targetEntryBase(entry) + params.coarse.w], 0u);
  }
}

@compute @workgroup_size(128)
fn assembleGalerkin(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= sourceCount()) { return; }
  if (!fineActive(row)) {
    for (var entry = sourceRowStart(row); entry < sourceRowStart(row + 1u); entry += 1u) {
      if (sourceColumn(entry) == row) {
        let targetEntry = entryParent(entry);
        atomicAddFloat(&targetMatrix[targetEntryBase(targetEntry) + params.coarse.w], 1.0);
        break;
      }
    }
    return;
  }
  for (var entry = sourceRowStart(row); entry < sourceRowStart(row + 1u); entry += 1u) {
    let targetEntry = entryParent(entry);
    atomicAddFloat(&targetMatrix[targetEntryBase(targetEntry) + params.coarse.w], sourceCoefficient(entry));
  }
}

@compute @workgroup_size(128)
fn copyFineResidual(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= sourceCount() || !solveActive()) { return; }
  setSourceF(RHS, row, bitcast<f32>(fineState[2u * sourceCount() + row]));
  setSourceF(CORRECTION, row, 0.0);
}

fn smoothLine(line: u32, add: bool) {
  let first = topology[params.topologyOffsets.z + line];
  let end = topology[params.topologyOffsets.z + line + 1u];
  var previous = 0xffffffffu; var previousC = 0.0; var previousD = 0.0;
  for (var slot = first; slot < end; slot += 1u) {
    let row = topology[params.topologyOffsets.w + slot];
    var next = 0xffffffffu; if (slot + 1u < end) { next = topology[params.topologyOffsets.w + slot + 1u]; }
    let coefficients = sourceLineValues(row, previous, next);
    let denominator = max(params.solve.w, coefficients.y - coefficients.x * previousC);
    let cPrime = coefficients.z / denominator;
    var rightHandSide = select(sourceF(RHS, row), sourceF(DEFECT, row), add);
    if (!add && params.counts.y != 0u) {
      rightHandSide = bitcast<f32>(fineState[2u * sourceCount() + row]);
      setSourceF(RHS, row, rightHandSide);
    }
    let dPrime = (rightHandSide - coefficients.x * previousD) / denominator;
    setSourceF(CPRIME, row, cPrime); setSourceF(DPRIME, row, dPrime);
    previous = row; previousC = cPrime; previousD = dPrime;
  }
  var nextValue = 0.0;
  for (var reverse = end; reverse > first; reverse -= 1u) {
    let row = topology[params.topologyOffsets.w + reverse - 1u];
    let value = sourceF(DPRIME, row) - sourceF(CPRIME, row) * nextValue;
    let weighted = params.solve.y * value;
    setSourceF(CORRECTION, row, select(weighted, sourceF(CORRECTION, row) + weighted, add));
    nextValue = value;
  }
}

@compute @workgroup_size(128)
fn lineSmoothInitial(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < params.counts.x && solveActive()) { smoothLine(gid.x, false); }
}

@compute @workgroup_size(128)
fn computeDefect(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < sourceCount() && solveActive()) { setSourceF(DEFECT, row, sourceF(RHS, row) - sourceProduct(row)); }
}

@compute @workgroup_size(128)
fn restrictDefectGather(@builtin(global_invocation_id) gid: vec3u) {
  let coarseRow = gid.x; if (coarseRow >= targetCount() || !solveActive()) { return; }
  let first = topology[params.transferOffsets.x + coarseRow];
  let end = topology[params.transferOffsets.x + coarseRow + 1u];
  var value = 0.0;
  for (var slot = first; slot < end; slot += 1u) {
    let row = topology[params.transferOffsets.y + slot];
    value += sourceF(RHS, row) - sourceProduct(row);
  }
  setTargetF(RHS, coarseRow, value);
}

@compute @workgroup_size(128)
fn prolongateCorrection(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= sourceCount() || !solveActive()) { return; }
  setSourceF(CORRECTION, row, sourceF(CORRECTION, row) + targetF(CORRECTION, nodeParent(row)));
}

@compute @workgroup_size(128)
fn lineSmoothFinal(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < params.counts.x && solveActive()) { smoothLine(gid.x, true); }
}

@compute @workgroup_size(128)
fn solveCoarsest(@builtin(local_invocation_id) lid: vec3u) {
  if (!solveActive()) { return; }
  let row = lid.x; let count = sourceCount();
  if (row < count) { setSourceF(CORRECTION, row, 0.0); }
  storageBarrier(); workgroupBarrier();
  // A fixed zero-start Richardson/Jacobi polynomial is a linear symmetric
  // operator: every term has the form D^-1 A ... A D^-1. Keeping the entire
  // <=96-row coarse level in one workgroup supplies the global ordering that
  // ordinary multi-dispatch Jacobi would require without serial Cholesky.
  for (var iteration = 0u; iteration < 8u; iteration += 1u) {
    if (row < count) {
      let defect = sourceF(RHS, row) - sourceProduct(row);
      setSourceF(DEFECT, row, defect);
    }
    storageBarrier(); workgroupBarrier();
    if (row < count) {
      let diagonal = max(params.solve.w, sourceMatrixValue(row, row));
      setSourceF(CORRECTION, row, sourceF(CORRECTION, row) + params.solve.y * sourceF(DEFECT, row) / diagonal);
    }
    storageBarrier(); workgroupBarrier();
  }
}

@compute @workgroup_size(128)
fn copyFineCorrection(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < sourceCount() && solveActive()) { fineState[4u * sourceCount() + row] = bitcast<u32>(sourceF(CORRECTION, row)); }
}
`;

export const quadtreeDispatchShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> scalars: array<f32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn updateDispatch() {
  let keepSolving = !(scalars[3] > 0.0 && scalars[2] <= bitcast<f32>(args[24]) * scalars[3]);
  if (keepSolving) { scalars[9] += 1.0; }
  scalars[10] = select(0.0, 1.0, keepSolving);
  for (var word = 0u; word < 12u; word += 1u) { args[word] = select(0u, args[12u + word], keepSolving || word % 3u != 0u); }
}
`;

/** Five-sweep narrow-band air-velocity extrapolation used after projection. */
export const quadtreeVelocityExtrapolationShader = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var levelSetIn: texture_3d<f32>;

fn valid(q: vec3i) -> bool { return all(q >= vec3i(0)) && all(q < vec3i(params.dims.xyz)); }
fn phi(q: vec3i) -> f32 { return select(3.402823e38, textureLoad(levelSetIn, q, 0).x, valid(q)); }
fn faceKnown(q: vec3i, axis: u32) -> bool {
  var plus = q; plus[axis] += 1;
  if (!valid(q) || !valid(plus)) { return false; }
  let inherited = bitcast<u32>(textureLoad(velocityIn, q, 0).w);
  return (inherited & (1u << axis)) != 0u || phi(q) < 0.0 || phi(plus) < 0.0;
}
@compute @workgroup_size(4, 4, 4)
fn extrapolateVelocity(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let q = vec3i(gid); var sample = textureLoad(velocityIn, q, 0); var value = sample.xyz;
  var knownMask = bitcast<u32>(sample.w) & 7u;
  let h = min(params.cell.x, min(params.cell.y, params.cell.z));
  let offsets = array<vec3i, 6>(vec3i(-1, 0, 0), vec3i(1, 0, 0), vec3i(0, -1, 0), vec3i(0, 1, 0), vec3i(0, 0, -1), vec3i(0, 0, 1));
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var plus = q; plus[axis] += 1;
    if (!valid(plus)) { value[axis] = 0.0; continue; }
    if (faceKnown(q, axis)) { knownMask |= 1u << axis; continue; }
    let ownPhi = phi(q); let otherPhi = phi(plus);
    if (!(ownPhi > 0.0 && otherPhi > 0.0 && min(ownPhi, otherPhi) < 5.0 * h)) { continue; }
    var sum = 0.0; var count = 0.0;
    for (var neighbour = 0u; neighbour < 6u; neighbour += 1u) {
      let n = q + offsets[neighbour];
      if (faceKnown(n, axis)) { sum += textureLoad(velocityIn, n, 0)[axis]; count += 1.0; }
    }
    if (count > 0.0) { value[axis] = sum / count; knownMask |= 1u << axis; }
  }
  textureStore(velocityOut, q, vec4f(value, bitcast<f32>(knownMask)));
}
`;

/** Dense diagnostic sampled by the scientific divergence slice. */
export const quadtreeDivergenceShader = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var divergenceOut: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
fn component(q: vec3i, axis: u32) -> f32 {
  if (any(q < vec3i(0)) || any(q >= vec3i(params.dims.xyz))) { return 0.0; }
  return textureLoad(velocityIn, q, 0)[axis];
}
@compute @workgroup_size(4, 4, 4)
fn computeDivergence(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let q = vec3i(gid);
  let value = (component(q + vec3i(1, 0, 0), 0u) - component(q, 0u)) / params.cell.x
    + (component(q + vec3i(0, 1, 0), 1u) - component(q, 1u)) / params.cell.y
    + (component(q + vec3i(0, 0, 1), 2u) - component(q, 2u)) / params.cell.z;
  textureStore(divergenceOut, q, vec4f(value));
}
`;

/** Last-resort current-step CFL guard with an explicit intervention counter. */
export const quadtreeVelocityClampShader = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> debugCounters: array<atomic<u32>>;
@compute @workgroup_size(4, 4, 4)
fn clampVelocity(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  var value = textureLoad(velocityIn, vec3i(gid), 0);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let limit = 0.9 * params.cell[axis] / max(params.cell.w, 1e-6);
    if (abs(value[axis]) > limit) { value[axis] = clamp(value[axis], -limit, limit); atomicAdd(&debugCounters[0], 1u); }
  }
  let speedLimit = 0.9 * min(params.cell.x, min(params.cell.y, params.cell.z)) / max(params.cell.w, 1e-6);
  let speed = length(value.xyz);
  if (speed > speedLimit) { value = vec4f(value.xyz * speedLimit / speed, value.w); atomicAdd(&debugCounters[0], 1u); }
  textureStore(velocityOut, vec3i(gid), value);
}
`;

/**
 * Flux-consistent transport-velocity view for the resident level set.
 *
 * The momentum write-back deliberately keeps fine sub-face deviations on
 * multi-subface variational faces (they carry real shear; box-filtering them
 * away costs a large KE deficit). Those deviations, however, are only
 * divergence-free at coarse granularity: advecting phi through them locally
 * compresses/expands the interface and steadily inflates its volume. This
 * pass rebuilds, for exactly those sub-faces, the face-mean sample the
 * variational solve actually constrained — the same quantity the momentum
 * write-back used before the shear-preserving change — into a separate
 * texture consumed only by the surface pipeline, never by momentum.
 */
export const quadtreeSurfaceTransportShader = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f }
struct Face { nodes: vec4u, coefficients: vec4f, bounds: vec4u, packed: u32, solidFlux: f32, weights: vec2f, sampleCells: vec4u, sampleSpans: vec4u, flux: f32, mlsMean: f32, volume: f32 }
@group(0) @binding(0) var velocityFinal: texture_3d<f32>;
@group(0) @binding(1) var transportOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var velocityPre: texture_3d<f32>;
@group(0) @binding(4) var cellProjection: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> faces: array<Face>;
@group(0) @binding(6) var levelSetIn: texture_3d<f32>;
fn faceSpan(face: Face) -> u32 { return face.packed & 0xffffu; }
fn faceAxis(face: Face) -> u32 { return (face.packed >> 16u) & 0x3u; }
fn faceSubfaceCount(face: Face) -> u32 {
  let ySpan = face.bounds.w - face.bounds.z;
  return select(faceSpan(face) * ySpan, faceSpan(face) * faceSpan(face) * ySpan, faceAxis(face) == 1u);
}
@compute @workgroup_size(4, 4, 4)
fn buildSurfaceTransport(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let id = vec3i(gid);
  var value = textureLoad(velocityFinal, id, 0);
  let projection = textureLoad(cellProjection, id, 0);
  let pre = textureLoad(velocityPre, id, 0).xyz;
  let ownPhi = textureLoad(levelSetIn, id, 0).x;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var plus = gid; plus[axis] += 1u;
    if (plus[axis] >= params.dims[axis]) { continue; }
    // Faces with both sides in air keep the narrow-band extrapolated value;
    // their pre-projection integrals never constrained anything.
    if (min(ownPhi, textureLoad(levelSetIn, vec3i(plus), 0).x) >= 0.0) { continue; }
    let packedFace = u32(round(projection[axis]));
    if (packedFace == 0u) { continue; }
    let face = faces[packedFace - 1u];
    if (faceSubfaceCount(face) > 1u && face.weights.x > 0.0) {
      // The projected momentum sample is (face mean + deviation); refreshFaces
      // integrated face.flux from the same pre-projection field, so removing
      // the pre-projection deviation leaves the flux-consistent face mean for
      // the level-set backtrace while momentum keeps its shear.
      value[axis] += (face.flux - face.solidFlux) / face.weights.x - pre[axis];
    }
  }
  textureStore(transportOut, id, value);
}
`;

interface GPUQuadtreeMultigridLevel {
  bindGroup: GPUBindGroup;
  rowGroups: number;
}

interface GPUQuadtreeMultigrid {
  hierarchy: QuadtreeMultigridHierarchy;
  levels: GPUQuadtreeMultigridLevel[];
  buffers: GPUBuffer[];
}

const quadtreeMultigridPipelineNames = [
  "clearCoarseMatrix", "assembleGalerkin", "copyFineResidual",
  "lineSmoothInitial", "computeDefect", "restrictDefectGather",
  "solveCoarsest", "prolongateCorrection", "lineSmoothFinal", "copyFineCorrection",
] as const;

export class WebGPUQuadtreeTallCellProjection {
  readonly info: QuadtreeTallCellProjectionInfo;
  private readonly buffers: GPUBuffer[];
  private readonly bindGroup: GPUBindGroup;
  private readonly mapPressureBindGroup: GPUBindGroup;
  private warmStartEnabled = false;
  private solveFeedbackReadback?: GPUBuffer;
  private solveFeedbackPending = false;
  private readonly solverFinalizeBindGroup: GPUBindGroup;
  private pipelines: Record<string, GPUComputePipeline>;
  private readonly shaderModule:GPUShaderModule;
  private readonly pipelineLayout:GPUPipelineLayout;
  private readonly gpuCache: QuadtreeGPUCache;
  private readonly params: GPUBuffer;
  private readonly scalarBuffer: GPUBuffer;
  private readonly dispatchArgs: GPUBuffer;
  private readonly dispatchBindGroup: GPUBindGroup;
  private readonly dispatchPipeline: GPUComputePipeline;
  private readonly velocityExtrapolationPipeline: GPUComputePipeline;
  private readonly extrapolateOutToScratchGroup: GPUBindGroup;
  private readonly extrapolateScratchToOutGroup: GPUBindGroup;
  private readonly cellProjection: GPUTexture;
  private readonly cellTopology: GPUTexture;
  private readonly factorAux: GPUTexture;
  private readonly cellPressureSamples: GPUTexture;
  private readonly mappedPressure: GPUTexture;
  private readonly mappedPressureStorageFallback: GPUTexture;
  private readonly mappedPressureSampledFallback: GPUTexture;
  private readonly divergence: GPUTexture;
  private readonly divergencePipeline: GPUComputePipeline;
  private readonly divergenceBindGroup: GPUBindGroup;
  private readonly velocityClampCounter: GPUBuffer;
  private readonly velocityClampPipeline: GPUComputePipeline;
  private readonly velocityClampBindGroup: GPUBindGroup;
  private readonly surfaceTransportPipeline: GPUComputePipeline;
  private readonly surfaceTransportBindGroup: GPUBindGroup;
  /** Mutable: the non-blocking inline-rebuild monitor refreshes it from packControl. */
  private dofCount: number;
  private readonly preconditionBlockGroups: number;
  private iterations: number;
  private iterationBudget: QuadtreeIterationBudget;
  private readonly parallelReductions: boolean;
  private readonly phaseQuerySet?: GPUQuerySet;
  private readonly phaseQueryResolve?: GPUBuffer;
  private readonly surfaceState: WebGPUQuadtreeSurfaceState;
  private readonly topologyWords: Uint32Array;
  private lastRelativeResidual?: number;
  levelSetMismatchFraction?: number;
  private readonly couplingBodyCount: number;
  private readonly couplingDistinctDofs: number;
  private readonly couplingBodyIndices: number[];
  private readonly displacedVolumes: number[];
  private lastResidualRms?: number;
  private lastInitialResidualRms?: number;
  private solveSequence = 0;
  private feedbackSequence = -1;
  /** Capacity-sized GPU pack targets bound by this projection; enables Algorithm-1 in-place rebuilds. */
  private readonly residentResources?: GPUQuadtreeResidentResources;
  private readonly inlineSupported: boolean;
  /** Last converged iteration count; absent until the ladder supplies a safe cost sample. */
  private megakernelIterationHint?: number;
  private readonly megakernelMode: QuadtreeMegakernelMode;
  private readonly megakernelEligible: boolean;
  private readonly pressureSolver: QuadtreePressureSolver;
  private multigrid?: GPUQuadtreeMultigrid;
  private multigridPipelines: Record<string, GPUComputePipeline> = {};
  /** Set by the monitor when a pack overflowed capacity; one async rebuild regrows and swaps. */
  inlineNeedsAsyncRebuild = false;
  private inlineMonitorBuffer?: GPUBuffer;
  private inlineMonitorPending = false;
  private inlineMonitorEncoded = false;
  private inlineExplicitSizing?: Float32Array;
  private inlineBuilder?: WebGPUQuadtreeBuilder;

  constructor(private readonly device: GPUDevice, private readonly scene: SceneDescription, private readonly dims: { nx: number; ny: number; nz: number }, private readonly resources: ProjectionResources, private readonly options: QuadtreeTallCellProjectionOptions, fields?: ProjectionFields, private readonly coupling?: QuadtreeRigidCoupling,deferPipelineCompilation=false,cache?:QuadtreeGPUCache) {
    const constructorStartedAt = performance.now();
    const { nx, ny, nz } = dims, h = { x: scene.container.width_m / nx, y: scene.container.height_m / ny, z: scene.container.depth_m / nz };
    const initial: ProjectionFields = fields ?? initialFields(scene, nx, ny, nz);
    if (!fields) initial.phi = signedDistanceFromVolume(Float32Array.from(initial.phi!, (value) => value < 0 ? 1 : 0), nx, ny, nz, h);
    if (!resources.levelSet && !initial.phi) throw new Error("Initial quadtree projection needs a level set");
    // The surface pipeline backtraces phi through a dedicated flux-consistent
    // transport view: the projected field with multi-subface deviations
    // replaced by their constrained face means (see
    // quadtreeSurfaceTransportShader). Momentum never reads this texture.
    const surfaceTransport = resources.surfaceTransport ?? (resources.surfaceTransport = device.createTexture({ label: "Quadtree surface transport velocity", size: [dims.nx, dims.ny, dims.nz], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    this.surfaceState = resources.levelSet ?? (resources.levelSet = new WebGPUQuadtreeSurfaceState(device, dims, h, surfaceTransport, initial.phi!, cache?.surface, resources.volume, options.debrisCulling === true, options.vofReconciliation === true));
    const tallGridStartedAt = performance.now();
    let pressureGrid: TallPressureGrid | undefined;
    let leafCount: number, pressureSampleCount: number, maximumNeighborRatio: number;
    if (initial.prepared) {
      ({ leafCount, pressureSampleCount, maximumNeighborRatio } = initial.prepared);
      this.topologyWords = initial.topologyWords ?? initial.prepared.topologyWords;
    } else {
      const explicitSizing = initialSizing(scene, nx, nz, h);
      let quadtree = initial.quadtree;
      if (!quadtree) {
        if (!initial.velocity) throw new Error("Initial quadtree construction needs a velocity field");
        if (!initial.phi) throw new Error("Initial quadtree sizing needs a level set");
        const sizing = quadtreeSizingFromVelocityAndSurface(initial.phi, initial.velocity, nx, ny, nz, h, undefined, undefined, undefined, undefined, options.deepSpeedGradientScale ?? 1);
        for (let index = 0; index < sizing.length; index += 1) sizing[index] = Math.max(sizing[index], explicitSizing[index]);
        quadtree = buildQuadtree(sizing, nx, nz, { h: Math.min(h.x, h.z), maximumLeafSize: options.maximumLeafSize, adaptivityStrength: options.adaptivityStrength, smoothingDilations: 3 });
      }
      if (!initial.pressureGrid && !initial.phi) throw new Error("Initial tall-grid population needs a level set");
      const adaptiveOpticalLayer = options.opticalLayerMode === "adaptive-motion" && initial.velocity
        ? buildAdaptiveOpticalLayerField(initial.phi!, initial.velocity, nx, ny, nz, h, { alpha: options.opticalAlpha })
        : undefined;
      pressureGrid = initial.pressureGrid ?? populateTallPressureGrid(quadtree, initial.phi!, ny, h, 1, options.opticalDepthFraction, undefined, adaptiveOpticalLayer);
      this.topologyWords = initial.topologyWords ?? pressureTopologyWords(pressureGrid);
      leafCount = quadtree.leaves.length;
      pressureSampleCount = pressureGrid.samples.length;
      maximumNeighborRatio = quadtree.maximumNeighborRatio;
    }
    const tallGrid_ms = initial.prepared?.tallGrid_ms ?? performance.now() - tallGridStartedAt;
    let variationalAssembly_ms: number, systemPack_ms: number;
    let packed: ReturnType<typeof WebGPUQuadtreeTallCellProjection.packSystem>;
    let faceCount: number, ghostFaceCount: number, maximumSystemFluidScale: number, tallSegmentCount: number;
    if (initial.prepared) {
      if (!initial.prepared.packed) throw new Error("A reused topology cannot construct a new projection");
      packed = initial.prepared.packed;
      ({ displacedVolumes: this.displacedVolumes, dofCount: this.dofCount, faceCount, ghostFaceCount, maximumFluidScale: maximumSystemFluidScale, tallSegmentCount, variationalAssembly_ms, systemPack_ms } = initial.prepared);
    } else {
      const variationalStartedAt = performance.now();
      const solidFields = solidFieldsFromBodies(scene, coupling?.bodies ?? [], nx, ny, nz, h);
      const variationalBodies = coupling ? variationalBodiesFrom(scene, coupling) : [];
      const system = buildVariationalSystem(pressureGrid!, {
        velocity: initial.velocity,
        solidFraction: solidFields?.solidFraction, solidOwner: solidFields?.solidOwner, bodies: variationalBodies
      }, { assembleDense: false });
      variationalAssembly_ms = performance.now() - variationalStartedAt;
      this.displacedVolumes = displacedVolumesForGrid(pressureGrid!, initial.phi, solidFields, coupling?.bodies.length ?? 0, nx, ny, h);
      this.dofCount = system.liquidSampleIds.length;
      const packStartedAt = performance.now();
      packed = WebGPUQuadtreeTallCellProjection.packSystem(system, nx, ny, nz, coupling?.dynamic ? variationalBodies : [], options.preconditioner);
      systemPack_ms = performance.now() - packStartedAt;
      faceCount = system.faces.length;
      ghostFaceCount = system.faces.filter((face) => face.ghost).length;
      maximumSystemFluidScale = system.faces.reduce((maximum, face) => Math.max(maximum, face.fluidScale), 0);
      tallSegmentCount = pressureGrid!.segments.filter((segment) => segment.tall).length;
    }
    // The hard cap remains the paper-faithful worst case. Only the amount of
    // command stream encoded ahead of the tolerance stop follows recent solves.
    // A cold polynomial solve historically needs 55--75 iterations, while
    // the geometric hierarchy reaches the same tolerance in roughly a dozen.
    // Starting MG from the generic hard-budget-derived hint would encode
    // hundreds of dead V-cycles on its first frames—the exact dispatch tail
    // this preconditioner is meant to remove. Feedback still grows the budget
    // immediately if a difficult solve exhausts this conservative seed.
    const opticalConditioning = adaptiveOpticalLayerDefaults(ny, { alpha: options.opticalAlpha });
    // Removing optical cubes does not remove the long vertical pressure mode:
    // the resulting taller cells can retain the condition number of the fixed
    // grid even while DOF count drops by an order of magnitude. Scale only the
    // safety cap by dmax/dmin; convergence feedback still controls encoded work.
    const conditionedOptions = options.opticalLayerMode === "adaptive-motion"
      ? { ...options, iterationConditioningScale: Math.max(options.iterationConditioningScale ?? 1, opticalConditioning.maximumCells / opticalConditioning.minimumCells) }
      : options;
    const budgetOptions = (options.preconditioner ?? "ic0") === "mg"
      && options.iterationBudgetHint === undefined && options.iterationEmaHint === undefined
      ? { ...conditionedOptions, iterationBudgetHint: 24, iterationEmaHint: 16 }
      : conditionedOptions;
    this.iterationBudget = quadtreeIterationBudget(this.dofCount, budgetOptions);
    this.iterations = this.iterationBudget.encodedBudget;
    this.megakernelIterationHint = options.megakernelIterationHint;
    this.parallelReductions = this.dofCount >= 4096;
    // The product method passes Chebyshev explicitly. Keep direct low-level
    // construction source-compatible with the former exact PCG behavior.
    this.pressureSolver = options.pressureSolver ?? "pcg";
    const uploadStartedAt = performance.now();
    const resident = initial.prepared?.resident;
    const faces = resident?.faces ?? bufferWithData(device, "Quadtree tall-cell variational faces", packed.faces, GPUBufferUsage.STORAGE, 112);
    const rowOffsets = resident?.rowOffsets ?? bufferWithData(device, "Quadtree tall-cell row offsets", packed.rowOffsets);
    const rowEntries = resident?.rowEntries ?? bufferWithData(device, "Quadtree tall-cell row entries", packed.rowEntries, GPUBufferUsage.STORAGE, 8);
    const matrixBuffer = resident?.matrixBuffer ?? bufferWithData(device, "Quadtree tall-cell refreshed CSR matrix", packed.matrixWords, GPUBufferUsage.STORAGE, 4);
    const factorColumns = resident?.factorColumns ?? bufferWithData(device, "Quadtree tall-cell IC(0) columns", packed.factorColumns, GPUBufferUsage.STORAGE, 8);
    const factorEntries = resident?.factorEntries ?? bufferWithData(device, "Quadtree tall-cell IC(0) entries", packed.factorEntries, GPUBufferUsage.STORAGE, 8);
    const factorAuxTexels = Math.max(1, Math.ceil(packed.factorAuxWords.length / 4)), factorAuxWidth = resident?.factorAuxWidth ?? Math.min(2048, factorAuxTexels), factorAuxHeight = Math.ceil(factorAuxTexels / factorAuxWidth);
    this.factorAux = resident?.factorAux ?? device.createTexture({ label: "Quadtree tall-cell IC level data", size: [factorAuxWidth, factorAuxHeight], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    if (!resident) {
      const factorAuxRowBytes = factorAuxWidth * 16, factorAuxPitch = Math.ceil(factorAuxRowBytes / 256) * 256, factorAuxUpload = new Uint8Array(factorAuxPitch * factorAuxHeight), factorAuxSource = new Uint8Array(packed.factorAuxWords.buffer);
      for (let row = 0; row < factorAuxHeight; row += 1) factorAuxUpload.set(factorAuxSource.subarray(row * factorAuxRowBytes, Math.min(factorAuxSource.length, (row + 1) * factorAuxRowBytes)), row * factorAuxPitch);
      device.queue.writeTexture({ texture: this.factorAux }, factorAuxUpload, { bytesPerRow: factorAuxPitch, rowsPerImage: factorAuxHeight }, { width: factorAuxWidth, height: factorAuxHeight });
    }
    this.cellProjection = resident?.cellProjection ?? device.createTexture({ label: "Quadtree tall-cell projection field", size: [nx, ny, nz], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const projectionRowBytes = nx * 16, projectionPitch = Math.ceil(projectionRowBytes / 256) * 256;
    if (!resident) {
      const projectionBytes = new Uint8Array(projectionPitch * ny * nz), projectionSource = new Uint8Array(packed.cellProjection.buffer);
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) projectionBytes.set(projectionSource.subarray(projectionRowBytes * (y + ny * z), projectionRowBytes * (y + ny * z + 1)), projectionPitch * (y + ny * z));
      device.queue.writeTexture({ texture: this.cellProjection }, projectionBytes, { bytesPerRow: projectionPitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    }
    this.cellTopology = resident?.cellTopology ?? device.createTexture({ label: "Quadtree tall-cell debug topology", size: [nx, ny, nz], dimension: "3d", format: "rg32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    if (!resident) {
      const topologyRowBytes = nx * 8, topologyPitch = Math.ceil(topologyRowBytes / 256) * 256;
      const topologyBytes = new Uint8Array(topologyPitch * ny * nz), topologySource = new Uint8Array(packed.cellTopology.buffer);
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) topologyBytes.set(topologySource.subarray(topologyRowBytes * (y + ny * z), topologyRowBytes * (y + ny * z + 1)), topologyPitch * (y + ny * z));
      device.queue.writeTexture({ texture: this.cellTopology }, topologyBytes, { bytesPerRow: topologyPitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    }
    this.cellPressureSamples = resident?.cellPressureSamples ?? device.createTexture({ label: "Quadtree cubic pressure sample topology", size: [nx, ny, nz], dimension: "3d", format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    if (!resident) {
      const sampleBytes = new Uint8Array(projectionPitch * ny * nz), sampleSource = new Uint8Array(packed.cellPressureSamples.buffer);
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) sampleBytes.set(sampleSource.subarray(projectionRowBytes * (y + ny * z), projectionRowBytes * (y + ny * z + 1)), projectionPitch * (y + ny * z));
      device.queue.writeTexture({ texture: this.cellPressureSamples }, sampleBytes, { bytesPerRow: projectionPitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    }
    this.mappedPressure = device.createTexture({ label: "Quadtree transient cubical MLS pressure", size: [nx, ny, nz], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.mappedPressureStorageFallback = device.createTexture({ label: "Quadtree MLS storage fallback", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.STORAGE_BINDING });
    this.mappedPressureSampledFallback = device.createTexture({ label: "Quadtree MLS sampled fallback", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    this.divergence = device.createTexture({ label: "Quadtree post-projection divergence", size: [nx, ny, nz], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.velocityClampCounter = device.createBuffer({ label: "Quadtree CFL safety telemetry", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // Count-dependent solver state is sized by the pack capacity when the
    // projection owns resident GPU pack targets, so in-place topology rebuilds
    // can grow the DOF count without any reallocation or bind-group churn.
    const dofCapacity = resident?.capacities ? Math.max(this.dofCount, resident.capacities.dofCapacity) : this.dofCount;
    const state = device.createBuffer({ label: "Quadtree tall-cell PCG SoA state", size: Math.max(32, dofCapacity * 32), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // Words 0..11 are the CG scalars; words 12+ hold the per-body coupling
    // six-vectors (stride 8, up to 12 bodies).
    const rowGroups = Math.ceil(this.dofCount / 128), partialWords = 2 * Math.ceil(dofCapacity / 128);
    const scalars = device.createBuffer({ label: "Quadtree tall-cell CG scalars and partial reductions", size: 4 * (108 + partialWords), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.scalarBuffer = scalars;
    this.dispatchArgs = device.createBuffer({ label: "Quadtree tall-cell active dispatches", size: 136, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.couplingBodyCount = packed.couplingBodyCount; this.couplingDistinctDofs = packed.couplingDistinctDofs; this.couplingBodyIndices = packed.couplingBodyIndices;
    this.params = device.createBuffer({ label: "Quadtree tall-cell parameters", size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([nx, ny, nz, factorAuxWidth, 0, 0, 0, 0]).buffer);
    device.queue.writeBuffer(this.params, 16, new Float32Array([h.x, h.y, h.z, 0]).buffer);
    const encodedPressureIterations = this.pressureSolver === "chebyshev" ? quadtreeChebyshevPasses(options.pressureIterations) : this.iterations;
    device.queue.writeBuffer(this.params, 32, new Uint32Array([this.dofCount, faceCount, encodedPressureIterations, packed.factorLevelCount]).buffer);
    const solveParams = new ArrayBuffer(16); new Float32Array(solveParams)[0] = options.relativeTolerance ** 2;
    new Uint32Array(solveParams).set([packed.levelsOffset, packed.rowOffsetsOffset, packed.rowEntriesOffset], 1); device.queue.writeBuffer(this.params, 48, solveParams);
    device.queue.writeBuffer(this.params, 64, new Uint32Array([packed.couplingByBodyOffset, packed.couplingByDofOffset, packed.couplingTableOffset, packed.couplingBodyCount]).buffer);
    // couplingCounts.z packs the warm-start flag in bit 0 and the persistent
    // solver's uniform hard cap in bits 1..12. The ladder never reads the cap.
    this.warmStartEnabled = (options.pressureWarmStart ?? true) && (this.pressureSolver === "chebyshev" || (options.preconditioner ?? "ic0") === "poly" || (options.preconditioner ?? "ic0") === "jacobi" || (options.preconditioner ?? "ic0") === "mg");
    const megakernelControl = (Math.min(2048, this.iterationBudget.hardBudget) << 1) | (this.warmStartEnabled ? 1 : 0);
    device.queue.writeBuffer(this.params, 80, new Uint32Array([packed.couplingDistinctDofs, packed.dofSamplesBase, megakernelControl, rowGroups]).buffer);
    // precondition.xy carry the line tables or the blockic tables; only the
    // active preconditioner's kernels are ever dispatched, so the overlay is
    // unambiguous per projection.
    this.preconditionBlockGroups = Math.ceil(packed.blockCount / 64);
    device.queue.writeBuffer(this.params, 96, new Uint32Array([
      (options.preconditioner ?? "ic0") === "blockic" ? packed.blockTableOffset : packed.lineOffsetsBase,
      (options.preconditioner ?? "ic0") === "blockic" ? packed.blockCount : packed.lineDofsBase,
      packed.lineCount, (options.preconditioner ?? "ic0") === "jacobi" ? 1 : Math.max(2, Math.min(4, Math.round(options.polynomialDegree ?? 2)))
    ]).buffer);
    const inflowBoundary = scene.fluid.inflow ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [nx, ny, nz]) : undefined;
    device.queue.writeBuffer(this.params, 112, new Float32Array([
      h.x, h.y, h.z, 0,
      scene.container.width_m, scene.container.height_m, scene.container.depth_m, 0,
      inflowBoundary?.outletCenter_m.x ?? 0, inflowBoundary?.outletCenter_m.y ?? 0, inflowBoundary?.outletCenter_m.z ?? 0, scene.fluid.inflow?.radius_m ?? 0,
      scene.fluid.inflow?.velocity_m_s.x ?? 0, scene.fluid.inflow?.velocity_m_s.y ?? 0, scene.fluid.inflow?.velocity_m_s.z ?? 0, inflowBoundary?.apertureScale ?? 0,
      0, 0, 0, 0
    ]).buffer);
    const couplingGroups = Math.ceil(packed.couplingDistinctDofs / 128);
    // Four next-iteration triples (row groups, single workgroup, coupling
    // groups, blockic blocks), their template, the squared tolerance, two
    // never-zeroed setup triples (words 25-30), and an independent current-
    // iteration row triple (31-33). The residual finalizer can switch off the
    // next iteration without suppressing the direction update it just earned.
    const dispatchWords = new Uint32Array(34);
    dispatchWords.set([rowGroups, 1, 1, 1, 1, 1, couplingGroups, 1, 1, Math.ceil(packed.blockCount / 64), 1, 1]);
    dispatchWords.set(dispatchWords.subarray(0, 12), 12);
    dispatchWords[24] = new Uint32Array(new Float32Array([options.relativeTolerance ** 2]).buffer)[0];
    dispatchWords.set([rowGroups, 1, 1, Math.ceil(faceCount / 128), 1, 1], 25);
    dispatchWords.set([rowGroups, 1, 1], 31);
    device.queue.writeBuffer(this.dispatchArgs, 0, dispatchWords);
    let layout: GPUBindGroupLayout;
    if (cache) {
      layout = cache.layout; this.shaderModule = cache.module; this.pipelineLayout = cache.pipelineLayout; this.gpuCache = cache;
    } else {
      layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ...Array.from({ length: 2 }, (_, index) => ({ binding: index + 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "2d" } },
        { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
        { binding: 15, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 16, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
        { binding: 17, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
      ] });
      this.shaderModule = device.createShaderModule({ label: "Quadtree tall-cell variational PCG", code: quadtreeTallCellProjectionShader });
      void this.shaderModule.getCompilationInfo().then((result) => { for (const message of result.messages) if (message.type === "error") console.error(`Quadtree tall-cell WGSL ${message.lineNum}:${message.linePos} ${message.message}`); }).catch(()=>{/* Device loss is reported by the renderer. */});
      this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      this.gpuCache = { layout, module: this.shaderModule, pipelineLayout: this.pipelineLayout };
    }
    this.gpuCache.surface = this.surfaceState.cache;
    if (!this.gpuCache.dispatchLayout || !this.gpuCache.dispatchPipeline) {
      this.gpuCache.dispatchLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
      ] });
      const dispatchModule = device.createShaderModule({ label: "Quadtree convergence dispatch", code: quadtreeDispatchShader });
      const dispatchPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.gpuCache.dispatchLayout] });
      this.gpuCache.dispatchPipeline = device.createComputePipeline({ label: "Quadtree convergence dispatch", layout: dispatchPipelineLayout, compute: { module: dispatchModule, entryPoint: "updateDispatch" } });
    }
    this.dispatchPipeline = this.gpuCache.dispatchPipeline;
    this.dispatchBindGroup = device.createBindGroup({ layout: this.gpuCache.dispatchLayout, entries: [
      { binding: 0, resource: { buffer: scalars } }, { binding: 1, resource: { buffer: this.dispatchArgs } }
    ] });
    if (!this.gpuCache.velocityExtrapolationLayout || !this.gpuCache.velocityExtrapolationPipeline) {
      this.gpuCache.velocityExtrapolationLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
      ] });
      const extrapolationModule = device.createShaderModule({ label: "Quadtree narrow-band velocity extrapolation", code: quadtreeVelocityExtrapolationShader });
      const extrapolationPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.gpuCache.velocityExtrapolationLayout] });
      this.gpuCache.velocityExtrapolationPipeline = device.createComputePipeline({
        label: "Quadtree narrow-band velocity extrapolation",
        layout: extrapolationPipelineLayout,
        compute: { module: extrapolationModule, entryPoint: "extrapolateVelocity" }
      });
    }
    this.velocityExtrapolationPipeline = this.gpuCache.velocityExtrapolationPipeline;
    const extrapolationGroup = (input: GPUTexture, output: GPUTexture) => device.createBindGroup({ layout: this.gpuCache.velocityExtrapolationLayout!, entries: [
      { binding: 0, resource: input.createView() }, { binding: 1, resource: output.createView() },
      { binding: 2, resource: { buffer: this.params } }, { binding: 3, resource: this.surfaceState.texture.createView() }
    ] });
    this.extrapolateOutToScratchGroup = extrapolationGroup(resources.velocityOut, resources.velocityScratch);
    this.extrapolateScratchToOutGroup = extrapolationGroup(resources.velocityScratch, resources.velocityOut);
    if (!this.gpuCache.velocityClampLayout || !this.gpuCache.velocityClampPipeline) {
      this.gpuCache.velocityClampLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
      ] });
      const module = device.createShaderModule({ label: "Quadtree CFL safety clamp", code: quadtreeVelocityClampShader });
      this.gpuCache.velocityClampPipeline = device.createComputePipeline({ label: "Quadtree CFL safety clamp", layout: device.createPipelineLayout({ bindGroupLayouts: [this.gpuCache.velocityClampLayout] }), compute: { module, entryPoint: "clampVelocity" } });
    }
    this.velocityClampPipeline = this.gpuCache.velocityClampPipeline;
    this.velocityClampBindGroup = device.createBindGroup({ layout: this.gpuCache.velocityClampLayout, entries: [
      { binding: 0, resource: resources.velocityOut.createView() }, { binding: 1, resource: resources.velocityScratch.createView() },
      { binding: 2, resource: { buffer: this.params } }, { binding: 3, resource: { buffer: this.velocityClampCounter } }
    ] });
    if (!this.gpuCache.surfaceTransportLayout || !this.gpuCache.surfaceTransportPipeline) {
      this.gpuCache.surfaceTransportLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
      ] });
      const module = device.createShaderModule({ label: "Quadtree surface transport velocity", code: quadtreeSurfaceTransportShader });
      void module.getCompilationInfo().then((result) => { for (const message of result.messages) if (message.type === "error") console.error(`Quadtree surface transport WGSL ${message.lineNum}:${message.linePos} ${message.message}`); }).catch(() => { /* Device loss is reported by the renderer. */ });
      this.gpuCache.surfaceTransportPipeline = device.createComputePipeline({ label: "Quadtree surface transport velocity", layout: device.createPipelineLayout({ bindGroupLayouts: [this.gpuCache.surfaceTransportLayout] }), compute: { module, entryPoint: "buildSurfaceTransport" } });
    }
    this.surfaceTransportPipeline = this.gpuCache.surfaceTransportPipeline;
    this.surfaceTransportBindGroup = device.createBindGroup({ layout: this.gpuCache.surfaceTransportLayout, entries: [
      { binding: 0, resource: resources.velocityOut.createView() },
      { binding: 1, resource: surfaceTransport.createView() },
      { binding: 2, resource: { buffer: this.params } },
      { binding: 3, resource: resources.velocityIn.createView() },
      { binding: 4, resource: this.cellProjection.createView() },
      { binding: 5, resource: { buffer: faces } },
      { binding: 6, resource: this.surfaceState.texture.createView() }
    ] });
    if (!this.gpuCache.divergenceLayout || !this.gpuCache.divergencePipeline) {
      this.gpuCache.divergenceLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
      ] });
      const module = device.createShaderModule({ label: "Quadtree divergence diagnostic", code: quadtreeDivergenceShader });
      this.gpuCache.divergencePipeline = device.createComputePipeline({ label: "Quadtree divergence diagnostic", layout: device.createPipelineLayout({ bindGroupLayouts: [this.gpuCache.divergenceLayout] }), compute: { module, entryPoint: "computeDivergence" } });
    }
    this.divergencePipeline = this.gpuCache.divergencePipeline;
    this.divergenceBindGroup = device.createBindGroup({ layout: this.gpuCache.divergenceLayout, entries: [
      { binding: 0, resource: resources.velocityOut.createView() }, { binding: 1, resource: this.divergence.createView() }, { binding: 2, resource: { buffer: this.params } }
    ] });
    const names = ["refreshFaces", "refreshRows", "initializeChebyshev", "iterateChebyshevAB", "iterateChebyshevBA", "finishChebyshevFromPressure", "finishChebyshevFromBest", "reduceChebyshevResidual", "initialize", "initializeJacobiDirection", "initializePolynomialStart", "precondition", "preconditionBlockIC", "preconditionJacobi", "preconditionLine", "preconditionPolynomialStart", "preconditionPolynomialMultiply", "preconditionPolynomialUpdate", "preconditionPolynomialUpdateDirection", "startDirection", "reduceInitial", "reduceInitialNorm", "solveMegakernel", "multiply", "multiplyPartial", "applyStep", "applyStepPartial", "applyStepFinalize", "applyStepUpdate", "applyStepUpdateJacobi", "applyStepUpdatePolynomialStart", "applyStepAlphaUpdate", "applyStepAlphaUpdatePolynomialStart", "applyStepAlphaUpdateJacobiFinishPartial", "finishIteration", "finishIterationPartial", "preconditionPolynomialUpdateFinishPartial", "finishIterationFinalize", "finishIterationUpdate", "mapPressure", "refreshFaceMls", "project", "coupleReduce", "coupleApply", "coupleImpulse"];
    this.pipelines = this.gpuCache.pipelines ?? (deferPipelineCompilation ? {} : Object.fromEntries(names.map((entryPoint) => [entryPoint, device.createComputePipeline(this.pipelineDescriptor(entryPoint))])));
    if (!this.gpuCache.pipelines && !deferPipelineCompilation) this.gpuCache.pipelines = this.pipelines;
    const all = [faces, rowOffsets, rowEntries, matrixBuffer, state, scalars, factorColumns, factorEntries];
    const projectionEntries = (storagePressure: GPUTexture, sampledPressure: GPUTexture, factorColumnsOverride = factorColumns): GPUBindGroupEntry[] => [
      { binding: 0, resource: resources.velocityIn.createView() }, { binding: 1, resource: resources.velocityOut.createView() },
      ...all.slice(0, 4).map((buffer, index) => ({ binding: index + 3, resource: { buffer } })),
      { binding: 7, resource: this.cellProjection.createView() },
      ...[state, scalars, factorColumnsOverride, factorEntries].map((buffer, index) => ({ binding: index + 8, resource: { buffer } })), { binding: 12, resource: { buffer: this.params } },
      { binding: 13, resource: this.factorAux.createView() },
      { binding: 14, resource: this.cellPressureSamples.createView() },
      { binding: 15, resource: this.surfaceState.texture.createView() },
      { binding: 16, resource: storagePressure.createView() },
      { binding: 17, resource: sampledPressure.createView() }
    ];
    this.bindGroup = device.createBindGroup({ layout, entries: projectionEntries(this.mappedPressureStorageFallback, this.mappedPressure) });
    this.mapPressureBindGroup = device.createBindGroup({ layout, entries: projectionEntries(this.mappedPressure, this.mappedPressureSampledFallback) });
    this.solverFinalizeBindGroup = device.createBindGroup({ layout, entries: projectionEntries(this.mappedPressureStorageFallback, this.mappedPressure, this.dispatchArgs) });
    const multigridBuffers: GPUBuffer[] = [];
    if (packed.multigrid) {
      if (!this.gpuCache.multigridLayout || !this.gpuCache.multigridModule || !this.gpuCache.multigridPipelineLayout) {
        this.gpuCache.multigridLayout = device.createBindGroupLayout({ entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ] });
        this.gpuCache.multigridModule = device.createShaderModule({ label: "Quadtree geometric multigrid", code: quadtreeMultigridShader });
        void this.gpuCache.multigridModule.getCompilationInfo().then((result) => { for (const message of result.messages) if (message.type === "error") console.error(`Quadtree multigrid WGSL ${message.lineNum}:${message.linePos} ${message.message}`); }).catch(() => { /* Device loss is reported by the renderer. */ });
        this.gpuCache.multigridPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.gpuCache.multigridLayout] });
      }
      this.multigridPipelines = this.gpuCache.multigridPipelines ?? (deferPipelineCompilation ? {} : Object.fromEntries(quadtreeMultigridPipelineNames.map((entryPoint) => [entryPoint, device.createComputePipeline({ label: `Quadtree multigrid ${entryPoint}`, layout: this.gpuCache.multigridPipelineLayout!, compute: { module: this.gpuCache.multigridModule!, entryPoint } })])));
      if (!this.gpuCache.multigridPipelines && !deferPipelineCompilation) this.gpuCache.multigridPipelines = this.multigridPipelines;

      const hierarchy = packed.multigrid;
      const matrixBuffers = hierarchy.levels.map((level, levelIndex) => {
        if (levelIndex === 0) return matrixBuffer;
        const buffer = bufferWithData(device, `Quadtree multigrid L${levelIndex} Galerkin matrix`, level.matrixWords!, GPUBufferUsage.STORAGE, 4);
        multigridBuffers.push(buffer); return buffer;
      });
      const vectorBuffers = hierarchy.levels.map((level, levelIndex) => {
        const buffer = device.createBuffer({ label: `Quadtree multigrid L${levelIndex} vectors`, size: Math.max(20, 5 * level.nodeCount * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        multigridBuffers.push(buffer); return buffer;
      });
      // WebGPU forbids overlapping writable bindings even when an entry point
      // never touches the second binding. The coarsest solve has no next
      // level, so bind distinct one-word sentinels instead of aliasing itself.
      const terminalMatrix = device.createBuffer({ label: "Quadtree multigrid terminal matrix sentinel", size: 4, usage: GPUBufferUsage.STORAGE });
      const terminalVectors = device.createBuffer({ label: "Quadtree multigrid terminal vector sentinel", size: 4, usage: GPUBufferUsage.STORAGE });
      multigridBuffers.push(terminalMatrix, terminalVectors);
      const levels = hierarchy.levels.map((level, levelIndex): GPUQuadtreeMultigridLevel => {
        const next = hierarchy.levels[levelIndex + 1] ?? level;
        const nodeMap = level.nodeToCoarse ?? new Uint32Array(0);
        const entryMap = level.entryToCoarse ?? new Uint32Array(0);
        const coarseNodeOffsets = level.coarseNodeOffsets ?? new Uint32Array(0);
        const coarseNodes = level.coarseNodes ?? new Uint32Array(0);
        const nodeMapOffset = 0;
        const entryMapOffset = nodeMap.length;
        const coarseNodeOffsetsOffset = entryMapOffset + entryMap.length;
        const coarseNodesOffset = coarseNodeOffsetsOffset + coarseNodeOffsets.length;
        const lineOffsetsOffset = coarseNodesOffset + coarseNodes.length;
        const lineNodesOffset = lineOffsetsOffset + level.lineOffsets.length;
        const topologyWords = new Uint32Array(lineNodesOffset + level.lineNodes.length);
        topologyWords.set(nodeMap, nodeMapOffset); topologyWords.set(entryMap, entryMapOffset);
        topologyWords.set(coarseNodeOffsets, coarseNodeOffsetsOffset); topologyWords.set(coarseNodes, coarseNodesOffset);
        topologyWords.set(level.lineOffsets, lineOffsetsOffset); topologyWords.set(level.lineNodes, lineNodesOffset);
        const topologyBuffer = bufferWithData(device, `Quadtree multigrid L${levelIndex} transfers`, topologyWords, GPUBufferUsage.STORAGE, 4);
        const uniformWords = new Uint32Array(24);
        uniformWords.set([level.nodeCount, level.columns.length, levelIndex === 0 ? 4 : 2, levelIndex === 0 ? 2 : 1], 0);
        uniformWords.set([next.nodeCount, next.columns.length, levelIndex + 1 < hierarchy.levels.length ? 2 : (levelIndex === 0 ? 4 : 2), levelIndex + 1 < hierarchy.levels.length ? 1 : (levelIndex === 0 ? 2 : 1)], 4);
        uniformWords.set([nodeMapOffset, entryMapOffset, lineOffsetsOffset, lineNodesOffset], 8);
        uniformWords.set([level.lineOffsets.length - 1, levelIndex === 0 ? 1 : 0, levelIndex, hierarchy.levels.length], 12);
        uniformWords.set([coarseNodeOffsetsOffset, coarseNodesOffset, 0, 0], 16);
        const uniformFloats = new Float32Array(uniformWords.buffer);
        uniformFloats.set([options.relativeTolerance ** 2, 0.7, 1e-6, 1e-12], 20);
        const uniform = bufferWithData(device, `Quadtree multigrid L${levelIndex} parameters`, uniformWords, GPUBufferUsage.UNIFORM, 96);
        multigridBuffers.push(topologyBuffer, uniform);
        const hasTarget = levelIndex + 1 < hierarchy.levels.length;
        const targetIndex = Math.min(levelIndex + 1, hierarchy.levels.length - 1);
        return {
          rowGroups: Math.ceil(level.nodeCount / 128),
          bindGroup: device.createBindGroup({ layout: this.gpuCache.multigridLayout!, entries: [
            { binding: 0, resource: { buffer: matrixBuffers[levelIndex] } },
            { binding: 1, resource: { buffer: hasTarget ? matrixBuffers[targetIndex] : terminalMatrix } },
            { binding: 2, resource: { buffer: vectorBuffers[levelIndex] } },
            { binding: 3, resource: { buffer: hasTarget ? vectorBuffers[targetIndex] : terminalVectors } },
            { binding: 4, resource: { buffer: topologyBuffer } },
            { binding: 5, resource: { buffer: scalars } },
            { binding: 7, resource: { buffer: state } },
            { binding: 8, resource: { buffer: uniform } },
          ] }),
        };
      });
      this.multigrid = { hierarchy, levels, buffers: multigridBuffers };
    }
    this.buffers = [...all, this.dispatchArgs, this.velocityClampCounter, ...multigridBuffers];
    // Algorithm-1 in-place rebuilds: available once the sparse pack lives in
    // capacity-sized GPU resources this projection owns. Rigid coupling and
    // incomplete-factor preconditioners stay on the asynchronous CPU path.
    const preconditionerChoice = options.preconditioner ?? "ic0";
    this.residentResources = resident;
    this.inlineSupported = !!resident?.packControl && !coupling && (preconditionerChoice === "poly" || preconditionerChoice === "jacobi");
    this.megakernelMode = options.megakernelSolve === false ? "off" : options.megakernelMode ?? "dynamic";
    this.megakernelEligible = this.pressureSolver === "pcg" && this.megakernelMode !== "off"
      && this.couplingBodyCount === 0
      && (preconditionerChoice === "poly" || preconditionerChoice === "jacobi");
    if (options.debugPressureTimings && device.features.has("timestamp-query")) {
      this.phaseQuerySet = device.createQuerySet({ label: "Quadtree pressure phase timings", type: "timestamp", count: 8 });
      this.phaseQueryResolve = device.createBuffer({ label: "Quadtree pressure phase timing resolve", size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    const allocatedBytes = this.buffers.reduce((sum, buffer) => sum + buffer.size, this.params.size + nx * ny * nz * 28);
    const resourceUpload_ms = performance.now() - uploadStartedAt;
    const opticalSettings = adaptiveOpticalLayerDefaults(ny, { alpha: options.opticalAlpha });
    this.info = { leafCount, pressureSampleCount, liquidDofCount: this.dofCount, faceCount, mlsProjectionRowCount: packed.mlsRowCount, tallSegmentCount, ghostFaceCount, maximumNeighborRatio, compressionRatio: this.dofCount / Math.max(1, nx * ny * nz), maximumFluidScale: maximumSystemFluidScale, opticalLayerMode: options.opticalLayerMode ?? "fixed", opticalAlpha: opticalSettings.alpha, opticalMinimumCells: opticalSettings.minimumCells, opticalMaximumCells: opticalSettings.maximumCells, allocatedBytes, cpuTallGrid_ms: tallGrid_ms, cpuVariationalAssembly_ms: variationalAssembly_ms, cpuSystemPack_ms: systemPack_ms, cpuICFactorization_ms: packed.icFactorization_ms, cpuResourceUpload_ms: resourceUpload_ms, cpuTopologyPack_ms: performance.now() - constructorStartedAt, topologyReused: false, topologyReuseCount: 0, pressureIterationBudget: encodedPressureIterations, pressureIterationHardBudget: this.pressureSolver === "chebyshev" ? encodedPressureIterations : this.iterationBudget.hardBudget, factorLevelCount: packed.factorLevelCount, multigridLevelCount: packed.multigrid?.levels.length, multigridCoarsestDofs: packed.multigrid?.coarsestNodeCount };
  }

  private pipelineDescriptor(entryPoint:string):GPUComputePipelineDescriptor{return{label:`Quadtree tall-cell ${entryPoint}`,layout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  static async createAsync(device:GPUDevice,scene:SceneDescription,dims:{nx:number;ny:number;nz:number},resources:ProjectionResources,options:QuadtreeTallCellProjectionOptions,fields?:ProjectionFields,coupling?:QuadtreeRigidCoupling,onProgress:(label:string,completed:number,total:number)=>void=()=>{},cache?:QuadtreeGPUCache){const projection=new WebGPUQuadtreeTallCellProjection(device,scene,dims,resources,options,fields,coupling,true,cache);await projection.initializePipelines(onProgress);return projection;}
  async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void=()=>{}){
    if(this.gpuCache.pipelines)this.pipelines=this.gpuCache.pipelines;
    else {const names=["refreshFaces","refreshRows","initializeChebyshev","iterateChebyshevAB","iterateChebyshevBA","finishChebyshevFromPressure","finishChebyshevFromBest","reduceChebyshevResidual","initialize","initializeJacobiDirection","initializePolynomialStart","precondition","preconditionBlockIC","preconditionJacobi","preconditionLine","preconditionPolynomialStart","preconditionPolynomialMultiply","preconditionPolynomialUpdate","preconditionPolynomialUpdateDirection","startDirection","reduceInitial","reduceInitialNorm","solveMegakernel","multiply","multiplyPartial","applyStep","applyStepPartial","applyStepFinalize","applyStepUpdate","applyStepUpdateJacobi","applyStepUpdatePolynomialStart","applyStepAlphaUpdate","applyStepAlphaUpdatePolynomialStart","applyStepAlphaUpdateJacobiFinishPartial","finishIteration","finishIterationPartial","preconditionPolynomialUpdateFinishPartial","finishIterationFinalize","finishIterationUpdate","mapPressure","refreshFaceMls","project","coupleReduce","coupleApply","coupleImpulse"];const pipelines:Record<string,GPUComputePipeline>={};for(let index=0;index<names.length;index+=1){const entryPoint=names[index];onProgress(`Adaptive pressure · ${entryPoint}`,index,names.length);pipelines[entryPoint]=await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint));onProgress(`Adaptive pressure · ${entryPoint}`,index+1,names.length);}this.pipelines=pipelines;this.gpuCache.pipelines=pipelines;}
    if(this.multigrid){
      if(this.gpuCache.multigridPipelines)this.multigridPipelines=this.gpuCache.multigridPipelines;
      else {const pipelines:Record<string,GPUComputePipeline>={};for(let index=0;index<quadtreeMultigridPipelineNames.length;index+=1){const entryPoint=quadtreeMultigridPipelineNames[index];onProgress(`Adaptive pressure multigrid · ${entryPoint}`,index,quadtreeMultigridPipelineNames.length);pipelines[entryPoint]=await this.device.createComputePipelineAsync({label:`Quadtree multigrid ${entryPoint}`,layout:this.gpuCache.multigridPipelineLayout!,compute:{module:this.gpuCache.multigridModule!,entryPoint}});onProgress(`Adaptive pressure multigrid · ${entryPoint}`,index+1,quadtreeMultigridPipelineNames.length);}this.multigridPipelines=pipelines;this.gpuCache.multigridPipelines=pipelines;}
    }
  }

  /**
   * Count-scan-emit reference for Appendix A's resident rebuild path.
   *
   * The runtime polynomial/Jacobi/line preconditioners do not consume an
   * incomplete factor.  For an uncoupled solve we can therefore emit the
   * exact face graph and both CSR streams directly from the segmented grid,
   * without first allocating a VariationalFace object graph and then walking
   * it again in packSystem.  The same three phases map one-for-one to the GPU
   * kernels; retaining this CPU implementation gives those kernels a compact
   * byte-exact oracle.
   */
  static packUncoupledGrid(grid: TallPressureGrid, nx: number, ny: number, nz: number, preconditioner: QuadtreeTallCellProjectionOptions["preconditioner"] = "poly") {
    const faceStride = 28, { quadtree, h } = grid, leafCount = quadtree.leaves.length;
    const dofBySample = new Int32Array(grid.samples.length); dofBySample.fill(-1);
    let dofCount = 0;
    for (const sample of grid.samples) if (sample.liquid) dofBySample[sample.id] = dofCount++;

    const segmentCounts = new Uint32Array(leafCount);
    for (const segment of grid.segments) segmentCounts[segment.leaf] += 1;
    const segmentStarts = new Uint32Array(leafCount + 1);
    for (let leaf = 0; leaf < leafCount; leaf += 1) segmentStarts[leaf + 1] = segmentStarts[leaf] + segmentCounts[leaf];

    type HorizontalVisitor = (axis: 0 | 2, x: number, z: number, transverseStart: number, transverseSpan: number, y0: number, y1: number, leftLeaf: number, rightLeaf: number) => void;
    const visitHorizontal = (axis: 0 | 2, visitor?: HorizontalVisitor) => {
      let count = 0;
      for (let z = 0; z < quadtree.nz; z += 1) for (let x = 0; x < quadtree.nx; x += 1) {
        const qx = x + (axis === 0 ? 1 : 0), qz = z + (axis === 2 ? 1 : 0);
        if (qx >= quadtree.nx || qz >= quadtree.nz) continue;
        const leftId = quadtree.leafAt[x + quadtree.nx * z], rightId = quadtree.leafAt[qx + quadtree.nx * qz];
        if (leftId === rightId) continue;
        const left = quadtree.leaves[leftId], right = quadtree.leaves[rightId];
        if (axis === 0 && x + 1 !== left.x + left.size) continue;
        if (axis === 2 && z + 1 !== left.z + left.size) continue;
        if (axis === 0 && z > 0 && quadtree.leafAt[x + quadtree.nx * (z - 1)] === leftId && quadtree.leafAt[qx + quadtree.nx * (z - 1)] === rightId) continue;
        if (axis === 2 && x > 0 && quadtree.leafAt[x - 1 + quadtree.nx * z] === leftId && quadtree.leafAt[x - 1 + quadtree.nx * qz] === rightId) continue;
        const transverseStart = axis === 0 ? Math.max(left.z, right.z) : Math.max(left.x, right.x);
        const transverseEnd = axis === 0 ? Math.min(left.z + left.size, right.z + right.size) : Math.min(left.x + left.size, right.x + right.size);
        let li = segmentStarts[leftId], ri = segmentStarts[rightId];
        const le = segmentStarts[leftId + 1], re = segmentStarts[rightId + 1];
        while (li < le && ri < re) {
          const leftSegment = grid.segments[li], rightSegment = grid.segments[ri];
          const y0 = Math.max(leftSegment.firstY, rightSegment.firstY), y1 = Math.min(leftSegment.lastY + 1, rightSegment.lastY + 1);
          if (y1 <= y0) { if (leftSegment.lastY < rightSegment.lastY) li += 1; else ri += 1; continue; }
          count += 1; visitor?.(axis, x, z, transverseStart, transverseEnd - transverseStart, y0, y1, leftId, rightId);
          if (leftSegment.lastY + 1 === y1) li += 1;
          if (rightSegment.lastY + 1 === y1) ri += 1;
        }
      }
      return count;
    };
    const horizontalFaceCount = visitHorizontal(0) + visitHorizontal(2);
    let verticalFaceCount = 0;
    for (const column of grid.samplesByLeaf) verticalFaceCount += Math.max(0, column.length - 1);
    const faceCount = horizontalFaceCount + verticalFaceCount;
    const faceBuffer = new ArrayBuffer(Math.max(1, faceCount * faceStride) * 4), faceU32 = new Uint32Array(faceBuffer), faceF32 = new Float32Array(faceBuffer);
    const cellCount = nx * ny * nz, cellProjection = new Float32Array(cellCount * 4), cellTopology = adaptivePressureCellTopology(grid);
    const cellPressureSamples = new Uint32Array(cellCount * 4); cellPressureSamples.fill(INVALID);
    let faceIndex = 0, ghostFaceCount = 0, maximumSystemFluidScale = 0;
    const sampleInterpolation = (column: TallPressureSample[], y: number) => {
      let lower = column[0], upper = column[column.length - 1];
      for (const sample of column) {
        if (sample.y <= y) lower = sample;
        if (sample.y >= y) { upper = sample; break; }
      }
      if (lower.id === upper.id) return [lower.id, INVALID, 1, 0] as const;
      const t = (y - lower.y) / Math.max(1, upper.y - lower.y);
      return [lower.id, upper.id, 1 - t, t] as const;
    };
    const emitFace = (axis: 0 | 1 | 2, boundsX: number, boundsZ: number, y0: number, y1: number, span: number, sampleIds: readonly number[], coefficients: readonly number[], volume: number, ghost: boolean) => {
      const offset = faceIndex * faceStride, nodeCount = sampleIds.length;
      let allPhi = 0, liquidPhi = 0, allLiquid = true;
      for (let slot = 0; slot < 4; slot += 1) {
        const sampleId = slot < nodeCount ? sampleIds[slot] : INVALID;
        const dof = sampleId === INVALID ? -1 : dofBySample[sampleId];
        faceU32[offset + slot] = dof < 0 ? INVALID : dof;
        faceF32[offset + 4 + slot] = slot < nodeCount ? coefficients[slot] : 0;
        if (sampleId !== INVALID) {
          const sample = grid.samples[sampleId], leaf = quadtree.leaves[sample.leaf], coefficient = coefficients[slot];
          faceU32[offset + 16 + slot] = leaf.x | (leaf.z << 10) | (sample.y << 20);
          faceU32[offset + 20 + slot] = leaf.size;
          allPhi += coefficient * sample.phi;
          if (sample.liquid) liquidPhi += coefficient * sample.phi; else allLiquid = false;
        }
      }
      const fluidScale = allLiquid ? 1 : Math.abs(liquidPhi) < 1e-12 ? 0 : Math.min(100, Math.max(0, allPhi / liquidPhi));
      maximumSystemFluidScale = Math.max(maximumSystemFluidScale, fluidScale);
      faceU32.set([boundsX, boundsZ, y0, y1], offset + 8);
      faceU32[offset + 12] = span | (axis << 16) | (nodeCount << 18) | ((ghost ? 1 : 0) << 21);
      faceF32[offset + 13] = 0; faceF32[offset + 14] = volume; faceF32[offset + 15] = volume * fluidScale; faceF32[offset + 26] = volume;
      if (axis === 0 || axis === 2) {
        for (let y = y0; y < y1; y += 1) for (let transverse = 0; transverse < span; transverse += 1) {
          const x = axis === 0 ? boundsX : boundsX + transverse, z = axis === 2 ? boundsZ : boundsZ + transverse;
          if (x < nx && z < nz) cellProjection[4 * (x + nx * (y + ny * z)) + axis] = faceIndex + 1;
        }
      } else {
        const leaf = quadtree.leaves[grid.samples[sampleIds[0]].leaf];
        for (let z = leaf.z; z < leaf.z + leaf.size && z < nz; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size && x < nx; x += 1) for (let y = y0; y < y1 && y < ny; y += 1) cellProjection[4 * (x + nx * (y + ny * z)) + 1] = faceIndex + 1;
      }
      if (ghost) ghostFaceCount += 1;
      faceIndex += 1;
    };
    const emitHorizontal: HorizontalVisitor = (axis, x, z, transverseStart, span, y0, y1, leftId, rightId) => {
      const left = quadtree.leaves[leftId], right = quadtree.leaves[rightId], distance = axis === 0 ? (left.size + right.size) * h.x / 2 : (left.size + right.size) * h.z / 2;
      const queryY = 0.5 * (y0 + y1) - 0.5;
      const l = sampleInterpolation(grid.samplesByLeaf[leftId], queryY), r = sampleInterpolation(grid.samplesByLeaf[rightId], queryY);
      const ids: number[] = [l[0]], coefficients: number[] = [-l[2] / distance];
      if (l[1] !== INVALID) { ids.push(l[1]); coefficients.push(-l[3] / distance); }
      ids.push(r[0]); coefficients.push(r[2] / distance);
      if (r[1] !== INVALID) { ids.push(r[1]); coefficients.push(r[3] / distance); }
      const volume = distance * (y1 - y0) * h.y * span * (axis === 0 ? h.z : h.x);
      emitFace(axis, axis === 0 ? x : transverseStart, axis === 2 ? z : transverseStart, y0, y1, span, ids, coefficients, volume, false);
    };
    visitHorizontal(0, emitHorizontal); visitHorizontal(2, emitHorizontal);
    for (const leaf of quadtree.leaves) {
      const column = grid.samplesByLeaf[leaf.id];
      for (let index = 0; index + 1 < column.length; index += 1) {
        const bottom = column[index], top = column[index + 1], distance = (top.y - bottom.y) * h.y;
        if (!(distance > 0)) continue;
        emitFace(1, leaf.x, leaf.z, bottom.y, top.y, leaf.size, [bottom.id, top.id], [-1 / distance, 1 / distance], distance * leaf.size * leaf.size * h.x * h.z, top.y - bottom.y > 1);
      }
    }
    if (faceIndex !== faceCount) throw new Error(`Direct variational face count mismatch: counted ${faceCount}, emitted ${faceIndex}`);

    const incidentCounts = new Uint32Array(dofCount), matrixCounts = new Uint32Array(dofCount);
    for (let face = 0; face < faceCount; face += 1) {
      const offset = face * faceStride, nodeCount = (faceU32[offset + 12] >> 18) & 7;
      let active = 0; for (let slot = 0; slot < nodeCount; slot += 1) if (faceU32[offset + slot] !== INVALID) active += 1;
      for (let slot = 0; slot < nodeCount; slot += 1) { const dof = faceU32[offset + slot]; if (dof !== INVALID) { incidentCounts[dof] += 1; matrixCounts[dof] += active; } }
    }
    const rowOffsets = new Uint32Array(dofCount + 1), matrixOffsets = new Uint32Array(dofCount + 1);
    for (let row = 0; row < dofCount; row += 1) { rowOffsets[row + 1] = rowOffsets[row] + incidentCounts[row]; matrixOffsets[row + 1] = matrixOffsets[row] + matrixCounts[row]; }
    const entryCount = rowOffsets[dofCount], rowEntriesBuffer = new ArrayBuffer(Math.max(1, entryCount) * 8), rowEntryU32 = new Uint32Array(rowEntriesBuffer), rowEntryF32 = new Float32Array(rowEntriesBuffer);
    const matrixEntryCount = matrixOffsets[dofCount], matrixWords = new Uint32Array(dofCount + 1 + 4 * matrixEntryCount), matrixFloats = new Float32Array(matrixWords.buffer);
    matrixWords.set(matrixOffsets, 0);
    const incidentCursor = rowOffsets.slice(0, dofCount), matrixCursor = matrixOffsets.slice(0, dofCount);
    for (let face = 0; face < faceCount; face += 1) {
      const offset = face * faceStride, nodeCount = (faceU32[offset + 12] >> 18) & 7;
      for (let a = 0; a < nodeCount; a += 1) {
        const row = faceU32[offset + a]; if (row === INVALID) continue;
        const incident = incidentCursor[row]++; rowEntryU32[2 * incident] = face; rowEntryF32[2 * incident + 1] = faceF32[offset + 4 + a];
        for (let b = 0; b < nodeCount; b += 1) {
          const column = faceU32[offset + b]; if (column === INVALID) continue;
          const entry = matrixCursor[row]++, base = dofCount + 1 + 4 * entry;
          matrixWords[base] = column; matrixWords[base + 1] = face | (b << 30);
          matrixFloats[base + 2] = 0; matrixFloats[base + 3] = faceF32[offset + 4 + a] * faceF32[offset + 4 + b];
        }
      }
    }

    let tallSegmentCount = 0;
    for (const segment of grid.segments) {
      const leaf = quadtree.leaves[segment.leaf], bottom = dofBySample[segment.bottomSample], top = dofBySample[segment.topSample];
      const packedY = segment.firstY | (segment.lastY << 10), packedLeaf = leaf.x | (leaf.z << 10) | (leaf.size << 20);
      if (segment.tall) tallSegmentCount += 1;
      for (let z = leaf.z; z < leaf.z + leaf.size && z < nz; z += 1) for (let y = segment.firstY; y <= segment.lastY; y += 1) for (let x = leaf.x; x < leaf.x + leaf.size && x < nx; x += 1) {
        const offset = 4 * (x + nx * (y + ny * z));
        cellPressureSamples[offset] = bottom < 0 ? INVALID : bottom; cellPressureSamples[offset + 1] = top < 0 ? INVALID : top;
        cellPressureSamples[offset + 2] = packedY; cellPressureSamples[offset + 3] = packedLeaf;
      }
    }
    for (const leaf of quadtree.leaves) {
      const column = grid.samplesByLeaf[leaf.id];
      for (let y = 0; y < ny; y += 1) {
        const [a, b, wa, wb] = interpolation(column, y), representedPhi = wa * grid.samples[a].phi + wb * grid.samples[b].phi;
        for (let z = leaf.z; z < leaf.z + leaf.size && z < nz; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size && x < nx; x += 1) cellProjection[4 * (x + nx * (y + ny * z)) + 3] = representedPhi < 0 ? leaf.size : -leaf.size;
      }
    }

    const lineCounts = new Uint32Array(leafCount); let lineCount = 0;
    for (const leaf of quadtree.leaves) { for (const sample of grid.samplesByLeaf[leaf.id]) if (dofBySample[sample.id] >= 0) lineCounts[leaf.id] += 1; if (lineCounts[leaf.id] > 0) lineCount += 1; }
    const levelsOffset = 0, rowOffsetsOffset = 0, rowEntriesOffset = dofCount + 1, blockTableOffset = rowEntriesOffset;
    // Keep the empty by-body and by-DOF CSR sentinels so offsets are byte-for-
    // byte identical to packSystem's zero-coupling layout.
    const couplingByBodyOffset = blockTableOffset, couplingByDofOffset = couplingByBodyOffset + 1, couplingTableOffset = couplingByDofOffset + 1;
    const dofSamplesBase = couplingTableOffset, lineOffsetsBase = dofSamplesBase + 4 * dofCount, lineDofsBase = lineOffsetsBase + lineCount + 1;
    const factorAuxWords = new Uint32Array(Math.max(4, lineDofsBase + dofCount));
    for (const sample of grid.samples) {
      const dof = dofBySample[sample.id]; if (dof < 0) continue;
      const leaf = quadtree.leaves[sample.leaf], segment = grid.segments[sample.segment], base = dofSamplesBase + 4 * dof;
      factorAuxWords[base] = leaf.x | (leaf.z << 10) | (sample.y << 20); factorAuxWords[base + 1] = leaf.size; factorAuxWords[base + 2] = segment.lastY - segment.firstY + 1;
    }
    let line = 0, lineDof = 0;
    for (const leaf of quadtree.leaves) {
      if (lineCounts[leaf.id] === 0) continue;
      factorAuxWords[lineOffsetsBase + line] = lineDof;
      for (const sample of grid.samplesByLeaf[leaf.id]) { const dof = dofBySample[sample.id]; if (dof >= 0) factorAuxWords[lineDofsBase + lineDof++] = dof; }
      line += 1;
    }
    factorAuxWords[lineOffsetsBase + lineCount] = lineDof;
    const factorColumns = new Uint8Array(Math.max(1, dofCount + 1) * 8);
    const multigrid = preconditioner === "mg"
      ? buildQuadtreeMultigridHierarchy(matrixWords, dofCount, factorAuxWords, dofSamplesBase, nx, ny, nz)
      : undefined;
    const packed = {
      faces: new Uint8Array(faceBuffer, 0, faceCount * faceStride * 4), rowOffsets, rowEntries: new Uint8Array(rowEntriesBuffer, 0, entryCount * 8), matrixWords,
      cellProjection, cellTopology, factorColumns, factorEntries: new Uint8Array(0), factorAuxWords,
      factorLevelCount: 1, levelsOffset, rowOffsetsOffset, rowEntriesOffset,
      couplingByBodyOffset, couplingByDofOffset, couplingTableOffset, couplingBodyCount: 0, couplingDistinctDofs: 0, couplingBodyIndices: [] as number[],
      dofSamplesBase, mlsRowCount: 0, cellPressureSamples, icFactorization_ms: 0,
      lineOffsetsBase, lineDofsBase, lineCount, blockTableOffset, blockCount: 0,
      ...(multigrid ? { multigrid } : {})
    };
    return { packed, dofCount, faceCount, ghostFaceCount, maximumFluidScale: maximumSystemFluidScale, tallSegmentCount };
  }

  static packSystem(system: VariationalSystem, nx: number, ny: number, nz: number, dynamicBodies: VariationalBody[], preconditioner: QuadtreeTallCellProjectionOptions["preconditioner"] = "ic0") {
    // The last two vec4s retain each pressure sample's leaf origin/y and span.
    // They let a reused topology refresh free-surface weights from the current
    // GPU level set without rebuilding or uploading the sparse face graph.
    // `flux` follows two vec4 fields, so WGSL's 16-byte struct alignment makes
    // the stride 28 words (112 bytes), not 25 words.
    const faceStride = 28, faces = new ArrayBuffer(system.faces.length * faceStride * 4), faceU32 = new Uint32Array(faces), faceF32 = new Float32Array(faces);
    const incident: Array<Array<{ face: number; coefficient: number }>> = Array.from({ length: system.liquidSampleIds.length }, () => []);
    const cellProjection = new Float32Array(nx * ny * nz * 4), cellTopology = adaptivePressureCellTopology(system.grid);
    // Per-cubic-cell adaptive pressure endpoints. The GPU uses this compact
    // topology to perform Narita Algorithm 1 line 10 on demand after CG,
    // avoiding materialized CPU MLS rows (paper Sec. 6's virtual split).
    const cellPressureSamples = new Uint32Array(nx * ny * nz * 4);
    cellPressureSamples.fill(INVALID);
    for (const segment of system.grid.segments) {
      const leaf = system.grid.quadtree.leaves[segment.leaf];
      const bottom = system.dofBySample[segment.bottomSample], top = system.dofBySample[segment.topSample];
      const packedY = segment.firstY | (segment.lastY << 10);
      const packedLeaf = leaf.x | (leaf.z << 10) | (leaf.size << 20);
      for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let y = segment.firstY; y <= segment.lastY; y += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) {
        const offset = 4 * (x + nx * (y + ny * z));
        cellPressureSamples[offset] = bottom < 0 ? INVALID : bottom;
        cellPressureSamples[offset + 1] = top < 0 ? INVALID : top;
        cellPressureSamples[offset + 2] = packedY;
        cellPressureSamples[offset + 3] = packedLeaf;
      }
    }
    const preconditionerChoice = preconditioner ?? "ic0", useBlockIC = preconditionerChoice === "blockic";
    // jacobi/line/poly read only the refreshed CSR diagonal, so both the
    // symbolic assembly below and the numeric factorization are skipped on
    // every topology rebuild when no incomplete-Cholesky factor is consumed.
    const buildIncompleteCholesky = preconditionerChoice === "ic0" || useBlockIC;
    const matrixRows: Array<Map<number, number>> = buildIncompleteCholesky
      ? Array.from({ length: system.liquidSampleIds.length }, () => new Map<number, number>())
      : [];
    system.faces.forEach((face, faceIndex) => {
      const offset = faceIndex * faceStride, nodeCount = face.nodes.length;
      for (let slot = 0; slot < 4; slot += 1) {
        const dof = slot < nodeCount ? system.dofBySample[face.nodes[slot]] : -1;
        faceU32[offset + slot] = dof < 0 ? INVALID : dof;
        faceF32[offset + 4 + slot] = slot < nodeCount ? face.coefficients[slot] : 0;
        if (slot < nodeCount) {
          const sample = system.grid.samples[face.nodes[slot]], leaf = system.grid.quadtree.leaves[sample.leaf];
          faceU32[offset + 16 + slot] = leaf.x | (leaf.z << 10) | (sample.y << 20);
          faceU32[offset + 20 + slot] = leaf.size;
        }
        if (dof >= 0) incident[dof].push({ face: faceIndex, coefficient: face.coefficients[slot] });
      }
      faceU32.set([face.bounds.x, face.bounds.z, face.bounds.y0, face.bounds.y1], offset + 8);
      faceU32[offset + 12] = face.bounds.span | (face.axis << 16) | (nodeCount << 18) | ((face.ghost ? 1 : 0) << 21);
      const va = face.volume * face.openFraction;
      faceF32[offset + 13] = face.volume * face.solidFlux;
      faceF32.set([va, va * face.fluidScale], offset + 14);
      faceF32[offset + 26] = face.volume;
      if (buildIncompleteCholesky) {
        const matrixWeight = va * face.fluidScale;
        const liquidTerms: Array<{ dof: number; coefficient: number }> = [];
        for (let slot = 0; slot < nodeCount; slot += 1) {
          const dof = system.dofBySample[face.nodes[slot]];
          if (dof >= 0) liquidTerms.push({ dof, coefficient: face.coefficients[slot] });
        }
        for (const a of liquidTerms) for (const b of liquidTerms) matrixRows[a.dof].set(b.dof, (matrixRows[a.dof].get(b.dof) ?? 0) + a.coefficient * b.coefficient * matrixWeight);
      }
      if (face.axis === 0 || face.axis === 2) {
        for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) for (let transverse = 0; transverse < face.bounds.span; transverse += 1) {
          const x = face.axis === 0 ? face.bounds.x : face.bounds.x + transverse;
          const z = face.axis === 2 ? face.bounds.z : face.bounds.z + transverse;
          if (x < nx && z < nz) cellProjection[4 * (x + nx * (y + ny * z)) + face.axis] = faceIndex + 1;
        }
      } else {
        const leaf = system.grid.quadtree.leaves[system.grid.samples[face.nodes[0]].leaf];
        for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) cellProjection[4 * (x + nx * (y + ny * z)) + 1] = faceIndex + 1;
      }
    });
    const rowOffsets = new Uint32Array(incident.length + 1), entryCount = incident.reduce((sum, row) => sum + row.length, 0), rowEntries = new ArrayBuffer(Math.max(1, entryCount) * 8), entryU32 = new Uint32Array(rowEntries), entryF32 = new Float32Array(rowEntries);
    let entry = 0; incident.forEach((row, index) => { rowOffsets[index] = entry; for (const item of row) { entryU32[2 * entry] = item.face; entryF32[2 * entry + 1] = item.coefficient; entry += 1; } }); rowOffsets[incident.length] = entry;
    // Per-step refresh turns these frozen face contributions into a flat CSR
    // matrix. Each row then performs dense f32/u32 streams with no face-graph
    // gather or four-slot walk inside the CG loop.
    let matrixEntryCount = 0;
    for (const row of incident) for (const item of row) for (const sample of system.faces[item.face].nodes) if (system.dofBySample[sample] >= 0) matrixEntryCount += 1;
    const matrixWords = new Uint32Array(incident.length + 1 + 4 * matrixEntryCount), matrixFloats = new Float32Array(matrixWords.buffer);
    let matrixEntry = 0;
    incident.forEach((row, rowIndex) => {
      matrixWords[rowIndex] = matrixEntry;
      for (const item of row) {
        const face = system.faces[item.face];
        for (let slot = 0; slot < face.nodes.length; slot += 1) {
          const node = system.dofBySample[face.nodes[slot]]; if (node < 0) continue;
          const base = incident.length + 1 + 4 * matrixEntry;
          matrixWords[base] = node; matrixWords[base + 1] = item.face | (slot << 30);
          matrixFloats[base + 2] = 0; matrixFloats[base + 3] = item.coefficient * face.coefficients[slot]; matrixEntry += 1;
        }
      }
    });
    matrixWords[incident.length] = matrixEntry;
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const leafId = system.grid.quadtree.leafAt[x + nx * z], leaf = system.grid.quadtree.leaves[leafId];
      for (let y = 0; y < ny; y += 1) {
        const [a, b, wa, wb] = interpolation(system.grid.samplesByLeaf[leafId], y), index = x + nx * (y + ny * z), dofA = system.dofBySample[a], dofB = system.dofBySample[b];
        void dofA; void dofB;
        const representedPhi = wa * system.grid.samples[a].phi + wb * system.grid.samples[b].phi;
        cellProjection[4 * index + 3] = representedPhi < 0 ? leaf.size : -leaf.size;
      }
    }
    const icStartedAt = performance.now();
    // Bridson's public-domain modified incomplete Cholesky level-zero
    // factorization (omega=0.97, minimum pivot ratio=0.25), generalized from
    // the regular seven-point stencil to this sparse variational matrix.
    const n = system.liquidSampleIds.length, factorStarts = new Uint32Array(n + 1), factorRows: number[] = [], factorValues: number[] = [];
    const factorPositions: Array<Map<number, number>> = buildIncompleteCholesky ? Array.from({ length: n }, () => new Map<number, number>()) : [];
    const originalDiagonal = new Float64Array(buildIncompleteCholesky ? n : 0), workDiagonal = new Float64Array(buildIncompleteCholesky ? n : 0), inverseDiagonal = new Float64Array(buildIncompleteCholesky ? n : 0);
    // blockic partitions the DOFs into blocks of whole leaf columns (up to
    // ~64 rows each; a deeper single column still forms one block). DOF ids
    // follow sample ids, which populateTallPressureGrid assigns leaf by leaf
    // and bottom-to-top, so every column is a contiguous ascending DOF range
    // and blocks of consecutive columns are contiguous row ranges. The factor
    // drops couplings that cross a block boundary, making each block's
    // triangular solves independent (one GPU lane per block).
    const blockStarts: number[] = [];
    if (useBlockIC) {
      const targetBlockRows = 64;
      let blockRows = 0;
      for (const column of system.grid.samplesByLeaf) {
        let first = -1, rows = 0;
        for (const sample of column) { const dof = system.dofBySample[sample.id]; if (dof >= 0) { if (first < 0) first = dof; rows += 1; } }
        if (rows === 0) continue;
        if (blockRows === 0 || blockRows + rows > targetBlockRows) { blockStarts.push(first); blockRows = 0; }
        blockRows += rows;
      }
    }
    const blockCount = blockStarts.length;
    const blockOf = new Int32Array(useBlockIC ? n : 0);
    for (let block = 0; block < blockCount; block += 1) blockOf.fill(block, blockStarts[block], block + 1 < blockCount ? blockStarts[block + 1] : n);
    if (buildIncompleteCholesky) {
      for (let column = 0; column < n; column += 1) {
        factorStarts[column] = factorRows.length;
        const diagonal = matrixRows[column].get(column) ?? 0; originalDiagonal[column] = diagonal; workDiagonal[column] = diagonal;
        for (const row of [...matrixRows[column].keys()].filter((row) => row > column && (!useBlockIC || blockOf[row] === blockOf[column])).sort((a, b) => a - b)) {
          factorPositions[column].set(row, factorRows.length); factorRows.push(row); factorValues.push(matrixRows[column].get(row) ?? 0);
        }
      }
      factorStarts[n] = factorRows.length;
      for (let column = 0; column < n; column += 1) {
        if (!(originalDiagonal[column] > 0)) continue;
        const safePivot = Number.isFinite(workDiagonal[column]) && workDiagonal[column] >= 0.25 * originalDiagonal[column] ? workDiagonal[column] : originalDiagonal[column];
        inverseDiagonal[column] = 1 / Math.sqrt(Math.max(safePivot, 1e-30));
        for (let p = factorStarts[column]; p < factorStarts[column + 1]; p += 1) factorValues[p] *= inverseDiagonal[column];
        for (let p = factorStarts[column]; p < factorStarts[column + 1]; p += 1) {
          const targetColumn = factorRows[p], multiplier = factorValues[p]; let missing = 0;
          for (let a = factorStarts[column]; a < factorStarts[column + 1]; a += 1) {
            const targetRow = factorRows[a], source = factorValues[a];
            if (targetRow < targetColumn) {
              if (!matrixRows[targetColumn].has(targetRow)) missing += source;
            } else if (targetRow === targetColumn) {
              workDiagonal[targetColumn] -= multiplier * source;
            } else {
              const destination = factorPositions[targetColumn].get(targetRow);
              if (destination === undefined) missing += source;
              else factorValues[destination] -= multiplier * source;
            }
          }
          // Narita et al. report that Bridson's modified variant performs
          // poorly because this variational operator is not an M-matrix. Their
          // selected IC(0) path applies no dropped-fill compensation.
          workDiagonal[targetColumn] -= 0 * multiplier * missing;
        }
      }
    } else factorStarts[n] = 0;
    const factorColumnsBuffer = new ArrayBuffer(Math.max(1, n + 1) * 8), factorColumnsU32 = new Uint32Array(factorColumnsBuffer), factorColumnsF32 = new Float32Array(factorColumnsBuffer);
    for (let column = 0; column <= n; column += 1) factorColumnsU32[2 * column] = factorStarts[column];
    for (let column = 0; column < n; column += 1) factorColumnsF32[2 * column + 1] = inverseDiagonal[column] ?? 0;
    const factorEntriesBuffer = new ArrayBuffer(Math.max(1, factorRows.length) * 8), factorEntriesU32 = new Uint32Array(factorEntriesBuffer), factorEntriesF32 = new Float32Array(factorEntriesBuffer);
    factorRows.forEach((row, index) => { factorEntriesU32[2 * index] = row; factorEntriesF32[2 * index + 1] = factorValues[index]; });
    const rowFactors: Array<Array<{ column: number; value: number }>> = buildIncompleteCholesky ? Array.from({ length: n }, () => []) : [];
    if (buildIncompleteCholesky) for (let column = 0; column < n; column += 1) for (let entryIndex = factorStarts[column]; entryIndex < factorStarts[column + 1]; entryIndex += 1) rowFactors[factorRows[entryIndex]].push({ column, value: factorValues[entryIndex] });
    const forwardLevel = new Uint32Array(buildIncompleteCholesky ? n : 0), backwardLevel = new Uint32Array(buildIncompleteCholesky ? n : 0);
    if (buildIncompleteCholesky) for (let row = 0; row < n; row += 1) for (const factor of rowFactors[row]) forwardLevel[row] = Math.max(forwardLevel[row], forwardLevel[factor.column] + 1);
    if (buildIncompleteCholesky) for (let column = n - 1; column >= 0; column -= 1) for (let entryIndex = factorStarts[column]; entryIndex < factorStarts[column + 1]; entryIndex += 1) backwardLevel[column] = Math.max(backwardLevel[column], backwardLevel[factorRows[entryIndex]] + 1);
    let deepestLevel = 0;
    if (buildIncompleteCholesky) for (let row = 0; row < n; row += 1) deepestLevel = Math.max(deepestLevel, forwardLevel[row], backwardLevel[row]);
    // With cross-block factor edges dropped, the global forward/backward
    // levels are also each block's local levels; levelCount stays the reported
    // schedule depth for either preconditioner.
    const levelCount = Math.max(1, 1 + deepestLevel);
    const schedule = new Uint32Array(Math.max(1, buildIncompleteCholesky ? 2 * n : 1)); let scheduleOffset = 0;
    let levels = new Uint32Array(0), blockTable = new Uint32Array(0);
    if (useBlockIC) {
      // One [rowStart, rowEnd) header per block; the GPU lane that owns the
      // block substitutes its rows serially in natural order, which respects
      // the triangular dependencies without any level schedule.
      blockTable = new Uint32Array(2 * blockCount);
      for (let block = 0; block < blockCount; block += 1) blockTable.set([blockStarts[block], block + 1 < blockCount ? blockStarts[block + 1] : n], 2 * block);
    } else if (buildIncompleteCholesky) {
      const forwardByLevel: number[][] = Array.from({ length: levelCount }, () => []), backwardByLevel: number[][] = Array.from({ length: levelCount }, () => []);
      for (let row = 0; row < n; row += 1) { forwardByLevel[forwardLevel[row]].push(row); backwardByLevel[backwardLevel[row]].push(row); }
      levels = new Uint32Array(levelCount * 4);
      for (let level = 0; level < levelCount; level += 1) {
        levels[4 * level] = scheduleOffset; schedule.set(forwardByLevel[level], scheduleOffset); scheduleOffset += forwardByLevel[level].length; levels[4 * level + 1] = scheduleOffset;
      }
      for (let level = 0; level < levelCount; level += 1) {
        levels[4 * level + 2] = scheduleOffset; schedule.set(backwardByLevel[level], scheduleOffset); scheduleOffset += backwardByLevel[level].length; levels[4 * level + 3] = scheduleOffset;
      }
    }
    const factorRowOffsets = new Uint32Array(n + 1), factorRowEntriesBuffer = new ArrayBuffer(Math.max(1, factorRows.length) * 8), factorRowEntriesU32 = new Uint32Array(factorRowEntriesBuffer), factorRowEntriesF32 = new Float32Array(factorRowEntriesBuffer);
    let rowEntry = 0;
    if (buildIncompleteCholesky) for (let row = 0; row < n; row += 1) {
      factorRowOffsets[row] = rowEntry;
      for (const factor of rowFactors[row]) { factorRowEntriesU32[2 * rowEntry] = factor.column; factorRowEntriesF32[2 * rowEntry + 1] = factor.value; rowEntry += 1; }
    }
    factorRowOffsets[n] = rowEntry;
    const icFactorization_ms = performance.now() - icStartedAt;
    const levelsOffset = scheduleOffset, rowOffsetsOffset = levelsOffset + levels.length, rowEntriesOffset = rowOffsetsOffset + factorRowOffsets.length;
    const blockTableOffset = rowEntriesOffset + 2 * rowEntry;
    // Rank-6 body couplings ride in the same aux-words texture (the storage
    // binding budget is exhausted): a by-body CSR for K^T reductions, a
    // by-DOF CSR for race-free K applications, and the per-body generalized
    // inverse masses (rho/m, then rho R I^-1 R^T row-major).
    const couplings = dynamicBodies.length > 0 ? system.couplings : [];
    const couplingRowCount = couplings.reduce((sum, coupling) => sum + coupling.rows.size, 0);
    const couplingBodyCount = couplings.length;
    const byDof = new Map<number, Array<{ body: number; row: Float64Array }>>();
    couplings.forEach((coupling, slot) => {
      // `slot` is the packed body index used by the WGSL loops and the mass
      // table; empty couplings were filtered out upstream.
      for (const [dof, row] of coupling.rows) {
        let list = byDof.get(dof);
        if (!list) { list = []; byDof.set(dof, list); }
        list.push({ body: slot, row });
      }
    });
    const couplingDistinctDofs = byDof.size;
    const couplingByBodyOffset = blockTableOffset + blockTable.length;
    const couplingByDofOffset = couplingByBodyOffset + (couplingBodyCount + 1) + couplingRowCount * 8;
    const couplingTableOffset = couplingByDofOffset + couplingDistinctDofs + (couplingDistinctDofs + 1) + couplingRowCount * 8;
    // Pressure-sample geometry follows coupling data in the shared auxiliary
    // texture. `mapPressure` gathers these records and evaluates the fixed
    // 4x4 MLS system in registers, so no per-face/per-cell rows are packed.
    const dofSamplesBase = couplingTableOffset + couplingBodyCount * 12;
    const lineRows = system.grid.samplesByLeaf.map((column) => column.map((sample) => system.dofBySample[sample.id]).filter((dof) => dof >= 0)).filter((row) => row.length > 0);
    const lineOffsetsBase = dofSamplesBase + 4 * n;
    const lineDofsBase = lineOffsetsBase + lineRows.length + 1;
    const lineDofCount = lineRows.reduce((sum, row) => sum + row.length, 0);
    const totalWords = lineDofsBase + lineDofCount;
    const factorAuxWords = new Uint32Array(Math.max(4, totalWords));
    const factorAuxFloats = new Float32Array(factorAuxWords.buffer);
    factorAuxWords.set(schedule.subarray(0, scheduleOffset), 0); factorAuxWords.set(levels, levelsOffset); factorAuxWords.set(factorRowOffsets, rowOffsetsOffset);
    factorAuxWords.set(new Uint32Array(factorRowEntriesBuffer, 0, 2 * rowEntry), rowEntriesOffset);
    factorAuxWords.set(blockTable, blockTableOffset);
    if (couplingBodyCount > 0) {
      let entryCursor = 0;
      const entriesBase = couplingByBodyOffset + couplingBodyCount + 1;
      couplings.forEach((coupling, index) => {
        factorAuxWords[couplingByBodyOffset + index] = entryCursor;
        for (const [dof, row] of coupling.rows) {
          const base = entriesBase + entryCursor * 8;
          factorAuxWords[base] = dof;
          for (let component = 0; component < 6; component += 1) factorAuxFloats[base + 2 + component] = row[component];
          entryCursor += 1;
        }
      });
      factorAuxWords[couplingByBodyOffset + couplingBodyCount] = entryCursor;
      const dofIds = [...byDof.keys()];
      const startsBase = couplingByDofOffset + couplingDistinctDofs, dofEntriesBase = startsBase + couplingDistinctDofs + 1;
      let dofCursor = 0;
      dofIds.forEach((dof, index) => {
        factorAuxWords[couplingByDofOffset + index] = dof;
        factorAuxWords[startsBase + index] = dofCursor;
        for (const entry of byDof.get(dof)!) {
          const base = dofEntriesBase + dofCursor * 8;
          factorAuxWords[base] = entry.body;
          for (let component = 0; component < 6; component += 1) factorAuxFloats[base + 2 + component] = entry.row[component];
          dofCursor += 1;
        }
      });
      factorAuxWords[startsBase + couplingDistinctDofs] = dofCursor;
      couplings.forEach((coupling, index) => {
        const body = dynamicBodies[coupling.body], base = couplingTableOffset + index * 12;
        factorAuxFloats[base] = body.inverseMass;
        for (let component = 0; component < 9; component += 1) factorAuxFloats[base + 1 + component] = body.inverseInertia[component];
      });
    }
    for (const sampleId of system.liquidSampleIds) {
      const dof = system.dofBySample[sampleId], sample = system.grid.samples[sampleId];
      const leaf = system.grid.quadtree.leaves[sample.leaf], segment = system.grid.segments[sample.segment];
      const base = dofSamplesBase + 4 * dof;
      factorAuxWords[base] = leaf.x | (leaf.z << 10) | (sample.y << 20);
      factorAuxWords[base + 1] = leaf.size;
      factorAuxWords[base + 2] = segment.lastY - segment.firstY + 1;
      factorAuxWords[base + 3] = 0;
    }
    {
      let cursor = 0;
      lineRows.forEach((row, index) => { factorAuxWords[lineOffsetsBase + index] = cursor; factorAuxWords.set(row, lineDofsBase + cursor); cursor += row.length; });
      factorAuxWords[lineOffsetsBase + lineRows.length] = cursor;
    }
    const multigrid = preconditioner === "mg"
      ? buildQuadtreeMultigridHierarchy(matrixWords, n, factorAuxWords, dofSamplesBase, nx, ny, nz)
      : undefined;
    return {
      faces: new Uint8Array(faces), rowOffsets, rowEntries: new Uint8Array(rowEntries, 0, entryCount * 8), matrixWords, cellProjection, cellTopology,
      factorColumns: new Uint8Array(factorColumnsBuffer), factorEntries: new Uint8Array(factorEntriesBuffer, 0, factorRows.length * 8),
      factorAuxWords, factorLevelCount: levelCount, levelsOffset, rowOffsetsOffset, rowEntriesOffset,
      couplingByBodyOffset, couplingByDofOffset, couplingTableOffset, couplingBodyCount, couplingDistinctDofs,
      couplingBodyIndices: couplings.map((coupling) => coupling.body),
      dofSamplesBase, mlsRowCount: 0, cellPressureSamples, icFactorization_ms,
      lineOffsetsBase, lineDofsBase, lineCount: lineRows.length,
      blockTableOffset, blockCount,
      ...(multigrid ? { multigrid } : {})
    };
  }

  encode(encoder: GPUCommandEncoder, nx: number, ny: number, nz: number, timestampWrites?: GPUComputePassTimestampWrites) {
    this.solveSequence += 1;
    encoder.clearBuffer(this.scalarBuffer);
    encoder.clearBuffer(this.velocityClampCounter);
    const coupled = this.couplingBodyCount > 0;
    const chebyshev = this.pressureSolver === "chebyshev";
    // Throughput mode keeps K only for the post-solve impulse publication.
    // The uploaded solid velocity remains in b, but K M^-1 K^T does not
    // re-enter every polynomial pass.
    const exactCoupled = coupled && !chebyshev;
    const chebyshevPasses = quadtreeChebyshevPasses(this.options.pressureIterations);
    const megakernelDegree = (this.options.preconditioner ?? "ic0") === "jacobi"
      ? 1
      : Math.max(2, Math.min(4, Math.round(this.options.polynomialDegree ?? 2)));
    const megakernel = this.megakernelEligible && (this.megakernelMode === "always"
      || (this.megakernelMode === "dynamic" && quadtreeMegakernelPreferred(
        this.dofCount,
        this.megakernelIterationHint,
        megakernelDegree,
        this.options.megakernelDofLimit,
        this.options.megakernelRowIterationLimit
      )));
    const rowGroups = Math.ceil(this.dofCount / 128);
    const preconditioner = this.options.preconditioner ?? "ic0";
    const polynomialDegree = Math.max(2, Math.min(4, Math.round(this.options.polynomialDegree ?? 2)));
    const multigridDispatch = (pass: GPUComputePassEncoder, entry: string, levelIndex: number, indirect: boolean, workgroups: number, single = false) => {
      const multigrid = this.multigrid;
      if (!multigrid) throw new Error("Geometric multigrid hierarchy was not packed");
      pass.setPipeline(this.multigridPipelines[entry]); pass.setBindGroup(0, multigrid.levels[levelIndex].bindGroup);
      if (indirect) pass.dispatchWorkgroupsIndirect(this.dispatchArgs, single ? 12 : 0);
      else pass.dispatchWorkgroups(workgroups);
    };
    const assembleMultigrid = (pass: GPUComputePassEncoder) => {
      const multigrid = this.multigrid; if (!multigrid) return;
      for (let level = 0; level + 1 < multigrid.levels.length; level += 1) {
        multigridDispatch(pass, "clearCoarseMatrix", level, false, multigrid.levels[level + 1].rowGroups);
        multigridDispatch(pass, "assembleGalerkin", level, false, multigrid.levels[level].rowGroups);
      }
    };
    const applyMultigrid = (pass: GPUComputePassEncoder, indirect: boolean) => {
      const multigrid = this.multigrid;
      if (!multigrid) throw new Error("Geometric multigrid hierarchy was not packed");
      const last = multigrid.levels.length - 1;
      if (last === 0) multigridDispatch(pass, "copyFineResidual", 0, indirect, multigrid.levels[0].rowGroups);
      for (let level = 0; level < last; level += 1) {
        const groups = multigrid.levels[level].rowGroups;
        multigridDispatch(pass, "lineSmoothInitial", level, indirect, groups);
        multigridDispatch(pass, "restrictDefectGather", level, indirect, multigrid.levels[level + 1].rowGroups);
      }
      multigridDispatch(pass, "solveCoarsest", last, indirect, 1, true);
      for (let level = last - 1; level >= 0; level -= 1) {
        const groups = multigrid.levels[level].rowGroups;
        multigridDispatch(pass, "prolongateCorrection", level, indirect, groups);
        multigridDispatch(pass, "computeDefect", level, indirect, groups);
        multigridDispatch(pass, "lineSmoothFinal", level, indirect, groups);
      }
      multigridDispatch(pass, "copyFineCorrection", 0, indirect, multigrid.levels[0].rowGroups);
    };
    const directPrecondition = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, pass: GPUComputePassEncoder) => {
      if (preconditioner === "ic0") direct("precondition", 1);
      else if (preconditioner === "blockic") direct("preconditionBlockIC", this.preconditionBlockGroups);
      else if (preconditioner === "jacobi") direct("preconditionJacobi", rowGroups);
      else if (preconditioner === "line") direct("preconditionLine", rowGroups);
      else if (preconditioner === "mg") applyMultigrid(pass, false);
      else {
        direct("preconditionPolynomialStart", rowGroups);
        for (let degree = 1; degree < polynomialDegree; degree += 1) { direct("preconditionPolynomialMultiply", rowGroups); direct("preconditionPolynomialUpdate", rowGroups); }
      }
    };
    const indirectPrecondition = (indirect: (entry: string, offset: number) => void, pass: GPUComputePassEncoder) => {
      if (preconditioner === "ic0") indirect("precondition", 12);
      else if (preconditioner === "blockic") indirect("preconditionBlockIC", 36);
      else if (preconditioner === "jacobi") indirect("preconditionJacobi", 0);
      else if (preconditioner === "line") indirect("preconditionLine", 0);
      else if (preconditioner === "mg") applyMultigrid(pass, true);
      else {
        indirect("preconditionPolynomialStart", 0);
        for (let degree = 1; degree < polynomialDegree; degree += 1) { indirect("preconditionPolynomialMultiply", 0); indirect("preconditionPolynomialUpdate", 0); }
      }
    };
    const withPass = (writes: GPUComputePassTimestampWrites | undefined, encode: (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void, pass: GPUComputePassEncoder) => void) => {
      const pass = encoder.beginComputePass(writes ? { timestampWrites: writes } : undefined);
      const direct = (entry: string, workgroups: number, y = 1, z = 1) => {
        pass.setPipeline(this.pipelines[entry]);
        pass.setBindGroup(0, entry === "mapPressure" ? this.mapPressureBindGroup : entry === "finishIterationFinalize" ? this.solverFinalizeBindGroup : this.bindGroup);
        pass.dispatchWorkgroups(workgroups, y, z);
      };
      const indirect = (entry: string, offset: number) => { pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroupsIndirect(this.dispatchArgs, offset); };
      encode(direct, indirect, pass); pass.end();
    };
    // In-place GPU rebuilds make the counts GPU-authoritative: every
    // count-shaped setup dispatch reads the never-zeroed indirect triples
    // (word 25 row groups, word 28 face groups) patched from packControl.
    const gpuCounts = this.inlineSupported;
    const indirectParallelSetup = (indirect: (entry: string, offset: number) => void, afterInit?: () => void) => {
      if (preconditioner === "jacobi") { indirect("initializeJacobiDirection", 100); afterInit?.(); return; }
      indirect("initializePolynomialStart", 100);
      afterInit?.();
      for (let degree = 1; degree < polynomialDegree; degree += 1) {
        indirect("preconditionPolynomialMultiply", 100);
        indirect(degree + 1 === polynomialDegree ? "preconditionPolynomialUpdateDirection" : "preconditionPolynomialUpdate", 100);
      }
    };
    const directParallelSetup = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, afterInit?: () => void) => {
      if (preconditioner === "jacobi") { direct("initializeJacobiDirection", rowGroups); afterInit?.(); return; }
      direct("initializePolynomialStart", rowGroups);
      afterInit?.();
      for (let degree = 1; degree < polynomialDegree; degree += 1) {
        direct("preconditionPolynomialMultiply", rowGroups);
        direct(degree + 1 === polynomialDegree ? "preconditionPolynomialUpdateDirection" : "preconditionPolynomialUpdate", rowGroups);
      }
    };
    const warmStart = this.warmStartEnabled;
    const refreshSystem = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void) => {
      if (gpuCounts) { indirect("refreshFaces", 112); indirect("refreshRows", 100); }
      else { direct("refreshFaces", Math.ceil(this.info.faceCount / 128)); direct("refreshRows", rowGroups); }
    };
    const setup = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void, pass: GPUComputePassEncoder) => {
      // Geometry-dependent values are refreshed once, outside the CG loop.
      // With warm start, |b|^2 is reduced right after initialization, before
      // the preconditioner setup recycles the staging slot.
      const captureNorm = warmStart ? () => direct("reduceInitialNorm", 1) : undefined;
      refreshSystem(direct, indirect);
      if (preconditioner === "mg") assembleMultigrid(pass);
      if (gpuCounts) {
        indirectParallelSetup(indirect, captureNorm); direct("reduceInitial", 1);
        return;
      }
      if (preconditioner === "poly" || preconditioner === "jacobi") directParallelSetup(direct, captureNorm);
      else { direct("initialize", rowGroups); captureNorm?.(); directPrecondition(direct, pass); direct("startDirection", rowGroups); }
      direct("reduceInitial", 1);
    };
    const iterations = (first: number, end: number, direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void, pass: GPUComputePassEncoder) => {
      void direct;
      if (this.parallelReductions && first === 0) { pass.setPipeline(this.dispatchPipeline); pass.setBindGroup(0, this.dispatchBindGroup); pass.dispatchWorkgroups(1); }
      for (let iteration = first; iteration < end; iteration += 1) {
        if (!this.parallelReductions) { pass.setPipeline(this.dispatchPipeline); pass.setBindGroup(0, this.dispatchBindGroup); pass.dispatchWorkgroups(1); }
        if (this.parallelReductions && !exactCoupled) {
          // Uncoupled fused path: the d.Ad partial rides inside the SpMV and
          // every update workgroup re-reduces the partials for alpha, so an
          // iteration needs no dedicated dot-product finalize dispatches.
          indirect("multiplyPartial", 0);
          if (preconditioner === "jacobi") {
            indirect("applyStepAlphaUpdateJacobiFinishPartial", 0);
          } else if (preconditioner === "poly") {
            indirect("applyStepAlphaUpdatePolynomialStart", 0);
            for (let degree = 1; degree < polynomialDegree; degree += 1) {
              indirect("preconditionPolynomialMultiply", 0);
              indirect(degree + 1 === polynomialDegree ? "preconditionPolynomialUpdateFinishPartial" : "preconditionPolynomialUpdate", 0);
            }
          } else {
            indirect("applyStepAlphaUpdate", 0); indirectPrecondition(indirect, pass); indirect("finishIterationPartial", 0);
          }
          direct("finishIterationFinalize", 1); indirect("finishIterationUpdate", 124);
          continue;
        }
        indirect("multiply", 0);
        if (exactCoupled) { indirect("coupleReduce", 12); indirect("coupleApply", 24); }
        if (this.parallelReductions) {
          indirect("applyStepPartial", 0); indirect("applyStepFinalize", 12);
          if (preconditioner === "jacobi") {
            indirect("applyStepUpdateJacobi", 0);
            indirect("finishIterationPartial", 0);
          } else if (preconditioner === "poly") {
            indirect("applyStepUpdatePolynomialStart", 0);
            for (let degree = 1; degree < polynomialDegree; degree += 1) {
              indirect("preconditionPolynomialMultiply", 0);
              indirect(degree + 1 === polynomialDegree ? "preconditionPolynomialUpdateFinishPartial" : "preconditionPolynomialUpdate", 0);
            }
          } else {
            indirect("applyStepUpdate", 0); indirectPrecondition(indirect, pass); indirect("finishIterationPartial", 0);
          }
          direct("finishIterationFinalize", 1); indirect("finishIterationUpdate", 124);
        } else {
          indirect("applyStep", 12); indirectPrecondition(indirect, pass); indirect("finishIteration", 12);
        }
      }
    };
    const mapPressure = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void) => {
      direct("mapPressure", Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    };
    const project = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void) => {
      if (coupled) direct("coupleImpulse", 1);
      // Paper Sec. 6: virtually split adaptive cells only for pressure
      // interpolation, then discard the temporary cubical pressure field.
      if (gpuCounts) indirect("refreshFaceMls", 112);
      else direct("refreshFaceMls", Math.ceil(this.info.faceCount / 128));
      direct("project", Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    };
    const chebyshevSolve = (direct: (entry: string, workgroups: number, y?: number, z?: number) => void, indirect: (entry: string, offset: number) => void, includeRefresh: boolean) => {
      if (includeRefresh) refreshSystem(direct, indirect);
      const rows = (entry: string) => gpuCounts ? indirect(entry, 100) : direct(entry, rowGroups);
      rows("initializeChebyshev");
      for (let pass = 0; pass < chebyshevPasses; pass += 1) rows(pass % 2 === 0 ? "iterateChebyshevAB" : "iterateChebyshevBA");
      rows(chebyshevPasses % 2 === 0 ? "finishChebyshevFromPressure" : "finishChebyshevFromBest");
      direct("reduceChebyshevResidual", 1);
    };
    if (this.phaseQuerySet && this.phaseQueryResolve) {
      if (timestampWrites) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } }); marker.end(); }
      const writes = (beginningOfPassWriteIndex: number, endOfPassWriteIndex: number): GPUComputePassTimestampWrites => ({ querySet: this.phaseQuerySet!, beginningOfPassWriteIndex, endOfPassWriteIndex });
      const firstCount = Math.min(this.iterations, Math.max(1, Math.round(this.options.debugPressureFirstIterations ?? 8)));
      if (chebyshev) {
        withPass(writes(0, 1), (direct, indirect) => refreshSystem(direct, indirect));
        // The fixed polynomial has no convergence-dependent boundary. Keep
        // its complete row-parallel ladder in one timing window.
        withPass(writes(2, 3), (direct, indirect) => chebyshevSolve(direct, indirect, false));
        withPass(writes(4, 5), () => {});
      } else if (megakernel) {
        withPass(writes(0, 1), (direct, indirect) => refreshSystem(direct, indirect));
        // The persistent dispatch includes initialization and every CG
        // iteration, so it occupies the first-iterations timing window; the
        // remaining-iterations window intentionally stays empty.
        withPass(writes(2, 3), (direct) => direct("solveMegakernel", 1));
        withPass(writes(4, 5), () => {});
      } else {
        withPass(writes(0, 1), (direct, indirect, pass) => setup(direct, indirect, pass));
        withPass(writes(2, 3), (direct, indirect, pass) => iterations(0, firstCount, direct, indirect, pass));
        withPass(writes(4, 5), (direct, indirect, pass) => iterations(firstCount, this.iterations, direct, indirect, pass));
      }
      withPass({ querySet: this.phaseQuerySet, beginningOfPassWriteIndex: 6 }, (direct) => mapPressure(direct));
      withPass({ querySet: this.phaseQuerySet, endOfPassWriteIndex: 7 }, (direct, indirect) => project(direct, indirect));
      encoder.resolveQuerySet(this.phaseQuerySet, 0, 8, this.phaseQueryResolve, 0);
      if (timestampWrites) { const marker = encoder.beginComputePass({ timestampWrites: { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } }); marker.end(); }
    } else {
      // MLS materialization writes a transient texture, so WebGPU requires a
      // pass boundary before the conservative projection samples it.
      withPass(timestampWrites ? { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } : undefined,
        (direct, indirect, pass) => {
          if (chebyshev) chebyshevSolve(direct, indirect, true);
          else if (megakernel) { refreshSystem(direct, indirect); direct("solveMegakernel", 1); }
          else { setup(direct, indirect, pass); iterations(0, this.iterations, direct, indirect, pass); }
        });
      withPass(undefined, (direct) => mapPressure(direct));
      withPass(timestampWrites ? { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } : undefined,
        (direct, indirect) => project(direct, indirect));
    }
    // A pressure kick can be created inside this solve, after the frame's
    // proactive subdivision decision. Clamp only that last-resort overshoot
    // to CFL 0.9 and count every touched component for the debug HUD.
    {
      const pass = encoder.beginComputePass({ label: "Quadtree current-step CFL safety clamp" });
      pass.setPipeline(this.velocityClampPipeline); pass.setBindGroup(0, this.velocityClampBindGroup);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    }
    // Ando--Batty-style narrow-band extrapolation: each dispatch grows the
    // set of known face velocities by one 6-neighbour ring. Five rings cover
    // the aligned 5h keep-alive band, including fast thin floor sheets.
    // The clamp left the field in scratch, so the odd sweep count starts
    // scratch->out and finishes in the public velocity texture with no
    // full-texture copies on either side.
    for (let sweep = 0; sweep < 5; sweep += 1) {
      const pass = encoder.beginComputePass({ label: `Quadtree velocity extrapolation ${sweep + 1}/5` });
      pass.setPipeline(this.velocityExtrapolationPipeline);
      pass.setBindGroup(0, sweep % 2 === 0 ? this.extrapolateScratchToOutGroup : this.extrapolateOutToScratchGroup);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
      pass.end();
    }
    // The divergence texture only feeds the debug grid overlay; refreshing it
    // on a short cadence keeps the overlay honest without spending a dense
    // dispatch on every solve.
    if (this.solveSequence % 4 === 1) {
      const divergencePass = encoder.beginComputePass({ label: "Quadtree post-projection divergence diagnostic" });
      divergencePass.setPipeline(this.divergencePipeline); divergencePass.setBindGroup(0, this.divergenceBindGroup);
      divergencePass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); divergencePass.end();
    }
  }

  encodeSurface(encoder: GPUCommandEncoder, dt_s: number, inflow?: SurfaceInflowState, cflSafetyDt_s = dt_s, timestampWrites?: GPUComputePassTimestampWrites) {
    // The projection's own inflowTiming.x gates applyInflowVelocity in project.
    this.device.queue.writeBuffer(this.params, 176, new Float32Array([inflow?.strength ?? 0, 0, 0, 0]));
    // The safety limit uses the whole submitted frame interval so a spike
    // created in the final substep cannot cross a cell before the next frame.
    this.device.queue.writeBuffer(this.params, 28, new Float32Array([cflSafetyDt_s]));
    // Materialize the flux-consistent transport view from this solve's faces
    // before phi transport; the momentum field itself is left untouched.
    {
      const { nx, ny, nz } = this.dims;
      const pass = encoder.beginComputePass({ label: "Quadtree surface transport velocity", ...(timestampWrites ? { timestampWrites: { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } } : {}) });
      pass.setPipeline(this.surfaceTransportPipeline); pass.setBindGroup(0, this.surfaceTransportBindGroup);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
    }
    this.surfaceState.encode(encoder, dt_s, inflow, timestampWrites ? { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } : undefined);
  }
  async readSurfaceDiagnostics() { const diagnostics = await this.surfaceState.readVolumeDiagnostics(); this.levelSetMismatchFraction = diagnostics.mismatchFraction; return diagnostics; }
  get surfaceDiagnostics() { return this.surfaceState.volumeDiagnostics; }
  addSurfaceReferenceVolumeCells(cells: number) { this.surfaceState.addReferenceVolumeCells(cells); }

  /** Resident signed-distance field; texture identity is stable across topology rebuilds. */
  get levelSetTexture() { return this.surfaceState.texture; }
  get pressureTexture() { return this.mappedPressure; }
  get pressureSamplesTexture() { return this.cellPressureSamples; }
  get divergenceTexture() { return this.divergence; }

  get bodyImpulseReadbackBytes() { return this.couplingBodyCount * 8 * 4; }

  encodeBodyImpulseReadback(encoder: GPUCommandEncoder, pooledReadback?: GPUBuffer, destinationOffset = 0) {
    if (!this.coupling?.dynamic || this.couplingBodyCount === 0) return undefined;
    const bytes = this.bodyImpulseReadbackBytes;
    const readback = pooledReadback ?? this.device.createBuffer({ label: "Quadtree per-step rigid impulse readback", size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.scalarBuffer, 12 * 4, readback, destinationOffset, bytes);
    return readback;
  }

  decodeMappedBodyImpulseReadback(readback: GPUBuffer, sourceOffset = 0): QuadtreeBodyImpulse[] {
    const solve = new Float32Array(readback.getMappedRange(sourceOffset, this.bodyImpulseReadbackBytes)), rho = this.scene.fluid.density_kg_m3;
    return this.couplingBodyIndices.map((bodyIndex, slot) => {
      const base = slot * 8, body = this.coupling!.bodies[bodyIndex];
      return {
        bodyId: body?.description.id ?? `body-${bodyIndex}`,
        impulse_N_s: { x: -rho * solve[base], y: -rho * solve[base + 1], z: -rho * solve[base + 2] },
        angularImpulse_N_m_s: { x: -rho * solve[base + 3], y: -rho * solve[base + 4], z: -rho * solve[base + 5] },
        displacedVolume_m3: this.displacedVolumes[bodyIndex] ?? 0
      };
    });
  }

  async readBodyImpulseReadback(readback: GPUBuffer): Promise<QuadtreeBodyImpulse[]> {
    await readback.mapAsync(GPUMapMode.READ);
    try { return this.decodeMappedBodyImpulseReadback(readback); }
    finally { readback.unmap(); readback.destroy(); }
  }

  async readSolveDiagnostics() {
    // A freshly rebuilt projection may be swapped in after the preceding solve
    // but before stats are sampled. Its scalar buffer is intentionally blank;
    // retain the diagnostics carried over by rebuildFromState until it encodes.
    if (this.solveSequence === 0 || this.solveFeedbackPending) return;
    this.solveFeedbackPending = true;
    const phaseBytes = this.phaseQueryResolve ? 64 : 0;
    const readback = this.solveFeedbackReadback ??= this.device.createBuffer({ label: "Quadtree solve diagnostics", size: 56 + phaseBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.scalarBuffer, 0, readback, 0, 48);
    encoder.copyBufferToBuffer(this.velocityClampCounter, 0, readback, 48, 4);
    if (this.phaseQueryResolve) encoder.copyBufferToBuffer(this.phaseQueryResolve, 0, readback, 56, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const solve = new Float32Array(readback.getMappedRange(0, 48));
      this.applySolveFeedback(solve);
      this.lastRelativeResidual = Math.sqrt(Math.max(0, solve[7]) / Math.max(1e-30, solve[3]));
      this.lastResidualRms = Math.sqrt(Math.max(0, solve[7]) / Math.max(1, this.dofCount));
      this.lastInitialResidualRms = Math.sqrt(Math.max(0, solve[3]) / Math.max(1, this.dofCount));
      this.info.velocityClampCount = new Uint32Array(readback.getMappedRange(48, 4))[0];
      if (phaseBytes) {
        const times = new BigUint64Array(readback.getMappedRange(56, 64));
        this.info.pressurePhaseTimings = {
          setup_ms: Number(times[1] - times[0]) / 1e6,
          firstIterations_ms: Number(times[3] - times[2]) / 1e6,
          remainingIterations_ms: Number(times[5] - times[4]) / 1e6,
          project_ms: Number(times[7] - times[6]) / 1e6
        };
      }
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      this.solveFeedbackPending = false;
    }
  }

  private applySolveFeedback(solve: Float32Array) {
    const used = Math.round(solve[9] ?? 0);
    const converged = solve[3] > 0 && solve[2] <= this.options.relativeTolerance ** 2 * solve[3];
    this.info.pressureIterationsUsed = used; this.info.pressureConverged = converged;
    if (this.pressureSolver === "chebyshev") return;
    // Only a converged solve is a safe workload predictor. An exhausted
    // ladder budget may understate the work the hard-capped megakernel would
    // need, so clear the hint until convergence is observed again.
    this.megakernelIterationHint = converged ? used : undefined;
    if (this.feedbackSequence === this.solveSequence) return;
    this.feedbackSequence = this.solveSequence;
    const nextBudget = nextQuadtreeIterationBudget(this.iterationBudget, used, converged);
    // A V-cycle has many more dispatches than the polynomial preconditioner,
    // so its dead tail needs tighter headroom. Two iterations plus a 10%
    // EMA margin proved sufficient across the dam-break transition; an
    // exhausted budget still takes the generic immediate 2x recovery path.
    this.iterationBudget = (this.options.preconditioner ?? "ic0") === "mg" && converged
      ? { ...nextBudget, encodedBudget: Math.max(Math.min(8, nextBudget.hardBudget), Math.min(nextBudget.hardBudget, Math.ceil(Math.max(used + 2, 1.1 * nextBudget.ema + 2)))) }
      : nextBudget;
    this.iterations = this.iterationBudget.encodedBudget;
    this.info.pressureIterationBudget = this.iterations;
    this.info.pressureIterationHardBudget = this.iterationBudget.hardBudget;
    this.device.queue.writeBuffer(this.params, 40, new Uint32Array([this.iterations]));
  }

  private feedbackOptions(): QuadtreeTallCellProjectionOptions {
    return {
      ...this.options,
      iterationBudgetHint: this.iterations,
      iterationEmaHint: this.iterationBudget.ema,
      megakernelIterationHint: this.megakernelIterationHint
    };
  }

  /** True when Algorithm-1's every-step topology regeneration can run fully on the GPU. */
  get canEncodeInlineRebuild() {
    return this.inlineSupported && !this.inlineNeedsAsyncRebuild
      && !!this.residentResources && !!this.gpuCache.gpuPack?.canEncodeResident(this.residentResources);
  }

  /**
   * Narita Algorithm 1, fully GPU-resident: encode sizing/subdivision, the
   * sparse pack, and the in-place publish into this projection's live
   * resources ahead of the step's own kernels. No readback sits between the
   * rebuild and the solve, so a fresh topology lands every simulation step;
   * a small non-blocking packControl monitor keeps telemetry current and
   * requests one asynchronous rebuild when capacities overflow.
   */
  encodeInlineRebuild(encoder: GPUCommandEncoder): boolean {
    const resident = this.residentResources, gpuPack = this.gpuCache.gpuPack;
    if (!resident || !gpuPack || !this.canEncodeInlineRebuild) return false;
    const { nx, ny, nz } = this.dims;
    const h = { x: this.scene.container.width_m / nx, y: this.scene.container.height_m / ny, z: this.scene.container.depth_m / nz };
    this.inlineBuilder ??= new WebGPUQuadtreeBuilder(this.device, this.dims, h, this.options.maximumLeafSize, this.options.adaptivityStrength, 3, this.gpuCache.construction, this.options.deepSpeedGradientScale ?? 1, this.options.opticalLayerMode ?? "fixed", this.options.opticalAlpha ?? 0.5);
    this.gpuCache.construction = this.inlineBuilder.cache;
    // Uncoupled scenes have static explicit sizing (rigid-coupled rebuilds
    // stay on the asynchronous path), so the CPU field is computed once.
    this.inlineExplicitSizing ??= initialSizing(this.scene, nx, nz, h);
    const finalTopology = this.inlineBuilder.encodeConstruction(encoder, {
      velocity: this.resources.velocityOut, levelSet: this.surfaceState.texture,
      explicitSizing: this.inlineExplicitSizing, diagnosticBytes: 48 + 12 * 8 * 4
    });
    if (!gpuPack.encodeResidentPack(encoder, finalTopology, this.surfaceState.texture, resident, this.inlineBuilder.opticalLayerBuffer)) return false;
    // Publish the (fresh or retained) counts into the uniform parameters and
    // the never-zeroed indirect setup triples. finalizeControl only advances
    // words 0-4 on a valid pack, so these copies are always consistent.
    const control = resident.packControl;
    encoder.copyBufferToBuffer(control, 0, this.params, 32, 8);   // dofCount, faceCount
    encoder.copyBufferToBuffer(control, 8, this.params, 84, 4);   // dofSamplesBase
    encoder.copyBufferToBuffer(control, 12, this.params, 92, 4);  // rowGroups (partial reductions)
    encoder.copyBufferToBuffer(control, 12, this.dispatchArgs, 48, 4);   // CG template row groups
    encoder.copyBufferToBuffer(control, 12, this.dispatchArgs, 100, 4);  // setup row groups
    encoder.copyBufferToBuffer(control, 16, this.dispatchArgs, 112, 4);  // setup face groups
    encoder.copyBufferToBuffer(control, 12, this.dispatchArgs, 124, 4);  // current-iteration finish groups
    if (!this.inlineMonitorPending) {
      this.inlineMonitorBuffer ??= this.device.createBuffer({ label: "Quadtree inline rebuild monitor", size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyBufferToBuffer(control, 0, this.inlineMonitorBuffer, 0, 64);
      this.inlineMonitorEncoded = true;
    }
    return true;
  }

  /** Kick the non-blocking monitor readback after the step's queue submission. */
  finishInlineRebuild() {
    if (!this.inlineMonitorEncoded || !this.inlineMonitorBuffer) return;
    const buffer = this.inlineMonitorBuffer;
    this.inlineMonitorEncoded = false; this.inlineMonitorPending = true;
    void buffer.mapAsync(GPUMapMode.READ).then(() => {
      const words = new Uint32Array(buffer.getMappedRange()).slice(); buffer.unmap();
      this.applyInlineControl(words);
    }).catch(() => { /* Device loss is reported by the renderer. */ }).finally(() => { this.inlineMonitorPending = false; });
  }

  private applyInlineControl(words: Uint32Array) {
    if (words[5] !== 1) {
      // Overflowed pack: the previous consistent topology stayed live; grow
      // capacities through one asynchronous rebuild (the retrying path).
      this.inlineNeedsAsyncRebuild = true;
      return;
    }
    const { nx, ny, nz } = this.dims;
    this.dofCount = words[0];
    this.info.liquidDofCount = words[0];
    this.info.faceCount = words[1];
    this.info.pressureSampleCount = words[6];
    this.info.leafCount = words[7];
    this.info.tallSegmentCount = words[8];
    this.info.ghostFaceCount = words[9];
    this.info.compressionRatio = words[0] / Math.max(1, nx * ny * nz);
  }

  async rebuildFromState(bodies?: RigidBodyState[]) {
    const { nx, ny, nz } = this.dims;
    const scalarBytes = 48 + 12 * 8 * 4;
    const h = { x: this.scene.container.width_m / nx, y: this.scene.container.height_m / ny, z: this.scene.container.depth_m / nz };
    const activeBodies = bodies ?? this.coupling?.bodies;
    const builder = new WebGPUQuadtreeBuilder(this.device, this.dims, h, this.options.maximumLeafSize, this.options.adaptivityStrength, 3, this.gpuCache.construction, this.options.deepSpeedGradientScale ?? 1, this.options.opticalLayerMode ?? "fixed", this.options.opticalAlpha ?? 0.5);
    this.gpuCache.construction = builder.cache;
    const currentPreconditioner = this.options.preconditioner ?? "ic0";
    const gpuSparseRequested = !this.coupling && (currentPreconditioner === "poly" || currentPreconditioner === "jacobi" || currentPreconditioner === "mg");
    const builderInputs = {
      velocity: this.resources.velocityOut, levelSet: this.surfaceState.texture,
      explicitSizing: initialSizing(this.scene, nx, nz, h, activeBodies), diagnosticBuffer: this.scalarBuffer, diagnosticBytes: scalarBytes
    };
    let built = await builder.build({ ...builderInputs, readLeafProfiles: !gpuSparseRequested });
    this.applySolveFeedback(built.diagnostics);
    const nextOptions = this.feedbackOptions();
    const nextCoupling = this.coupling ? { bodies: activeBodies ?? this.coupling.bodies, dynamic: this.coupling.dynamic } : undefined;
    // Appendix A: uncoupled parallel-preconditioned solves keep segmentation,
    // face enumeration, and CSR emission on the GPU. The worker remains the
    // byte-layout oracle and handles rigid coupling / incomplete factors.
    const preconditioner = nextOptions.preconditioner ?? "ic0";
    let prepared: PreparedProjectionCPU | undefined;
    if (!nextCoupling && (preconditioner === "poly" || preconditioner === "jacobi" || preconditioner === "mg")) {
      this.gpuCache.gpuPack ??= new WebGPUQuadtreePackBuilder(this.device, this.dims, h, nextOptions.opticalDepthFraction, nextOptions.opticalLayerMode ?? "fixed");
      const gpuPacked = await this.gpuCache.gpuPack.build(built.packedCells, this.surfaceState.texture, {
        dofCount: this.dofCount, faceCount: this.info.faceCount, pressureSampleCount: this.info.pressureSampleCount
      }, preconditioner !== "mg", builder.opticalLayerBuffer);
      if (gpuPacked && preconditioner === "mg") {
        // MG cannot publish a changed hierarchy in place yet, but it can
        // reuse the GPU's segmentation/face/CSR pack. Read back that compact
        // pack and build only dyadic transfers/symbolic coarse CSR on the CPU;
        // this removes the expensive worker face-graph construction.
        const packed = gpuPacked.packed as typeof gpuPacked.packed & { multigrid?: QuadtreeMultigridHierarchy };
        packed.multigrid = buildQuadtreeMultigridHierarchy(packed.matrixWords, gpuPacked.dofCount, packed.factorAuxWords, packed.dofSamplesBase, nx, ny, nz);
      }
      if (gpuPacked) prepared = {
        leafCount: gpuPacked.leafCount, pressureSampleCount: gpuPacked.pressureSampleCount, maximumNeighborRatio: gpuPacked.maximumNeighborRatio,
        topologyWords: built.packedCells.slice(), topologyIdentityComplete: false, packed: gpuPacked.packed, resident: gpuPacked.resident,
        displacedVolumes: [], dofCount: gpuPacked.dofCount, faceCount: gpuPacked.faceCount, ghostFaceCount: gpuPacked.ghostFaceCount,
        maximumFluidScale: gpuPacked.maximumFluidScale, tallSegmentCount: gpuPacked.tallSegmentCount,
        quadtreeDecode_ms: 0, tallGrid_ms: 0, variationalAssembly_ms: 0, systemPack_ms: 0, icFactorization_ms: 0, gpuPack_ms: gpuPacked.gpuWall_ms
      };
    }
    if (!prepared && built.columnProfiles.length === 0) {
      // Capacity growth is retried inside the GPU packer. Reaching this branch
      // therefore means validation/device corruption; recover through the
      // reference worker after obtaining only the profiles it requires.
      built = await builder.build({ ...builderInputs, readLeafProfiles: true });
      this.applySolveFeedback(built.diagnostics);
    }
    prepared ??= await prepareQuadtreeProjectionInWorker(this.gpuCache, {
      scene: this.scene, dims: this.dims, options: nextOptions,
      packedCells: built.packedCells, columnProfiles: built.columnProfiles, opticalColumns: built.opticalColumns,
      reuseTopologyWords: nextCoupling ? undefined : this.topologyWords,
      coupling: nextCoupling
    });
    const topologyReadbackBytes = built.columnProfiles.byteLength + built.opticalColumns.byteLength + nx * nz * 4 + scalarBytes + 16;
    const reuseTopology = !this.coupling && prepared.topologyIdentityComplete !== false
      && (prepared.reusedTopology === true || sameWords(this.topologyWords, prepared.topologyWords));
    let next: WebGPUQuadtreeTallCellProjection;
    if (reuseTopology) {
      const uploadStartedAt = performance.now();
      this.info.cpuTallGrid_ms = prepared.tallGrid_ms;
      this.info.cpuVariationalAssembly_ms = 0;
      this.info.cpuSystemPack_ms = 0;
      this.info.cpuICFactorization_ms = prepared.icFactorization_ms;
      this.info.cpuResourceUpload_ms = performance.now() - uploadStartedAt;
      this.info.topologyReused = true;
      this.info.topologyReuseCount = (this.info.topologyReuseCount ?? 0) + 1;
      // Returning the same projection is the cache-hit signal consumed by the
      // owner; it must not retire the still-resident buffers after the swap.
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      next = this;
    } else {
      if (!prepared.packed) throw new Error("Changed topology rebuild did not return a complete projection pack");
      const fields: ProjectionFields = { topologyWords: prepared.topologyWords, prepared };
      next = this.gpuCache.pipelines
        ? new WebGPUQuadtreeTallCellProjection(this.device, this.scene, this.dims, this.resources, nextOptions, fields, nextCoupling, false, this.gpuCache)
        : await WebGPUQuadtreeTallCellProjection.createAsync(this.device, this.scene, this.dims, this.resources, nextOptions, fields, nextCoupling, undefined, this.gpuCache);
      next.info.cpuTallGrid_ms = prepared.tallGrid_ms;
      next.info.topologyReused = false;
      next.info.topologyReuseCount = this.info.topologyReuseCount ?? 0;
    }
    next.levelSetMismatchFraction = 0;
    next.info.gpuConstruction_ms = built.gpuWall_ms;
    next.info.gpuConstructionKernel_ms = built.gpuKernel_ms;
    next.info.gpuSparsePack_ms = prepared.gpuPack_ms;
    next.info.cpuRedistance_ms = 0;
    next.info.cpuQuadtreeDecode_ms = prepared.quadtreeDecode_ms;
    next.info.cpuVariationalAssembly_ms = reuseTopology ? 0 : prepared.variationalAssembly_ms;
    next.info.cpuSystemPack_ms = reuseTopology ? 0 : prepared.systemPack_ms;
    next.info.cpuICFactorization_ms = prepared.icFactorization_ms;
    next.info.cpuTopologyPack_ms = prepared.quadtreeDecode_ms + prepared.tallGrid_ms + prepared.variationalAssembly_ms + prepared.systemPack_ms;
    next.info.topologyReadbackBytes = topologyReadbackBytes;
    const solve = built.diagnostics;
    next.info.pressureIterationsUsed = Math.round(solve[9] ?? 0);
    next.info.pressureConverged = solve[3] > 0 && solve[2] <= this.options.relativeTolerance ** 2 * solve[3];
    next.lastRelativeResidual = Math.sqrt(Math.max(0, solve[7]) / Math.max(1e-30, solve[3]));
    next.lastResidualRms = Math.sqrt(Math.max(0, solve[7]) / Math.max(1, this.dofCount));
    next.lastInitialResidualRms = Math.sqrt(Math.max(0, solve[3]) / Math.max(1, this.dofCount));
    return next;
  }

  get relativeResidual() { return this.lastRelativeResidual; }
  get residualRms() { return this.lastResidualRms; }
  get initialResidualRms() { return this.lastInitialResidualRms; }
  get topologyTexture() { return this.cellTopology; }
  get preconditioner() { const value = this.options.preconditioner; return value === "blockic" || value === "jacobi" || value === "line" || value === "poly" || value === "mg" ? value : "ic0"; }
  get solver() { return this.pressureSolver; }

  destroySharedSurface() { this.surfaceState.destroy(); this.resources.surfaceTransport?.destroy(); this.resources.surfaceTransport = undefined; WebGPUQuadtreeBuilder.destroyCache(this.gpuCache.construction); this.gpuCache.gpuPack?.destroy(); this.gpuCache.gpuPack = undefined; this.gpuCache.cpuWorker?.terminate(); this.gpuCache.cpuWorker = undefined; }
  destroy() { for (const buffer of this.buffers) buffer.destroy(); this.params.destroy(); this.cellProjection.destroy(); this.cellTopology.destroy(); this.factorAux.destroy(); this.cellPressureSamples.destroy(); this.mappedPressure.destroy(); this.mappedPressureStorageFallback.destroy(); this.mappedPressureSampledFallback.destroy(); this.divergence.destroy(); this.phaseQuerySet?.destroy(); this.phaseQueryResolve?.destroy(); this.inlineMonitorBuffer?.destroy(); this.solveFeedbackReadback?.destroy(); }
}

export function prepareQuadtreeProjectionCPU(input: QuadtreeCPUPreparationInput): PreparedProjectionCPU {
  const { nx, ny, nz } = input.dims;
  const h = { x: input.scene.container.width_m / nx, y: input.scene.container.height_m / ny, z: input.scene.container.depth_m / nz };
  const decodeStartedAt = performance.now();
  const quadtree = quadtreeFromPackedCells(input.packedCells, nx, nz);
  const quadtreeDecode_ms = performance.now() - decodeStartedAt;
  const tallStartedAt = performance.now();
  const opticalDefaults = adaptiveOpticalLayerDefaults(ny, { alpha: input.options.opticalAlpha });
  const adaptiveOpticalLayer: AdaptiveOpticalLayerField | undefined = input.options.opticalLayerMode === "adaptive-motion" && input.opticalColumns?.length === nx * nz * 4
    ? { columns: input.opticalColumns, surfaceOffsetCells: opticalDefaults.surfaceOffsetCells, airborneCells: opticalDefaults.airborneCells }
    : undefined;
  const pressureGrid = populateTallPressureGridFromLeafProfiles(quadtree, input.columnProfiles, ny, h, input.options.opticalDepthFraction, adaptiveOpticalLayer);
  const topologyWords = pressureTopologyWords(pressureGrid);
  const tallGrid_ms = performance.now() - tallStartedAt;
  if (input.reuseTopologyWords && sameWords(input.reuseTopologyWords, topologyWords)) {
    return {
      topologyWords, reusedTopology: true,
      leafCount: quadtree.leaves.length,
      pressureSampleCount: pressureGrid.samples.length,
      maximumNeighborRatio: quadtree.maximumNeighborRatio,
      displacedVolumes: [], dofCount: 0, faceCount: 0, ghostFaceCount: 0,
      maximumFluidScale: 0, tallSegmentCount: pressureGrid.segments.filter((segment) => segment.tall).length,
      quadtreeDecode_ms, tallGrid_ms, variationalAssembly_ms: 0, systemPack_ms: 0, icFactorization_ms: 0
    };
  }
  const preconditioner = input.options.preconditioner ?? "ic0";
  if (!input.coupling && preconditioner !== "ic0" && preconditioner !== "blockic") {
    const packStartedAt = performance.now();
    const direct = WebGPUQuadtreeTallCellProjection.packUncoupledGrid(pressureGrid, nx, ny, nz, preconditioner);
    const systemPack_ms = performance.now() - packStartedAt;
    return {
      leafCount: quadtree.leaves.length,
      pressureSampleCount: pressureGrid.samples.length,
      maximumNeighborRatio: quadtree.maximumNeighborRatio,
      topologyWords, packed: direct.packed,
      displacedVolumes: [], dofCount: direct.dofCount, faceCount: direct.faceCount,
      ghostFaceCount: direct.ghostFaceCount, maximumFluidScale: direct.maximumFluidScale,
      tallSegmentCount: direct.tallSegmentCount,
      quadtreeDecode_ms, tallGrid_ms, variationalAssembly_ms: 0, systemPack_ms, icFactorization_ms: 0
    };
  }
  const assemblyStartedAt = performance.now();
  const solidFields = solidFieldsFromBodies(input.scene, input.coupling?.bodies ?? [], nx, ny, nz, h);
  const variationalBodies = input.coupling ? variationalBodiesFrom(input.scene, input.coupling) : [];
  const system = buildVariationalSystem(pressureGrid, {
    solidFraction: solidFields?.solidFraction, solidOwner: solidFields?.solidOwner, bodies: variationalBodies
  }, { assembleDense: false });
  const variationalAssembly_ms = performance.now() - assemblyStartedAt;
  const packStartedAt = performance.now();
  const packed = WebGPUQuadtreeTallCellProjection.packSystem(system, nx, ny, nz, input.coupling?.dynamic ? variationalBodies : [], input.options.preconditioner);
  const systemPack_ms = performance.now() - packStartedAt;
  return {
    leafCount: quadtree.leaves.length,
    pressureSampleCount: pressureGrid.samples.length,
    maximumNeighborRatio: quadtree.maximumNeighborRatio,
    topologyWords, packed,
    displacedVolumes: displacedVolumesForGrid(pressureGrid, undefined, solidFields, input.coupling?.bodies.length ?? 0, nx, ny, h),
    dofCount: system.liquidSampleIds.length,
    faceCount: system.faces.length,
    ghostFaceCount: system.faces.filter((face) => face.ghost).length,
    maximumFluidScale: system.faces.reduce((maximum, face) => Math.max(maximum, face.fluidScale), 0),
    tallSegmentCount: pressureGrid.segments.filter((segment) => segment.tall).length,
    quadtreeDecode_ms, tallGrid_ms, variationalAssembly_ms, systemPack_ms, icFactorization_ms: packed.icFactorization_ms
  };
}

async function prepareQuadtreeProjectionInWorker(cache: QuadtreeGPUCache, input: QuadtreeCPUPreparationInput) {
  if (!cache.cpuWorker) {
    cache.cpuWorkerSequence = 0; cache.cpuWorkerPending = new Map();
    const receive = (data: { id: number; value?: PreparedProjectionCPU; error?: string }) => {
      const pending = cache.cpuWorkerPending?.get(data.id); if (!pending) return;
      cache.cpuWorkerPending!.delete(data.id);
      if (data.value) pending.resolve(data.value); else pending.reject(new Error(data.error ?? "Quadtree CPU worker failed"));
    };
    const fail = (message: string) => {
      for (const pending of cache.cpuWorkerPending?.values() ?? []) pending.reject(new Error(message));
      cache.cpuWorkerPending?.clear();
    };
    if (typeof Worker !== "undefined") {
      const worker = new Worker(new URL("./quadtree-topology-worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (event: MessageEvent<{ id: number; value?: PreparedProjectionCPU; error?: string }>) => receive(event.data);
      worker.onerror = (event) => fail(event.message);
      cache.cpuWorker = { postMessage: (message, transfer) => worker.postMessage(message, transfer ?? []), terminate: () => worker.terminate() };
    } else if (typeof process !== "undefined" && process.versions?.node) {
      // Keep Node smoke timings honest: topology packing must not synchronously
      // stall the driver thread while GPU timings are being sampled.
      const workerThreadsSpecifier = "node:" + "worker_threads";
      const { Worker: NodeWorker } = await import(/* @vite-ignore */ workerThreadsSpecifier) as typeof import("node:worker_threads");
      // Node 25's built-in strip-only loader rejects parameter properties in
      // imported application code. The tiny data-module entry uses tsx's
      // programmatic API, then loads the real worker with full TS transforms.
      const entryUrl = new URL("./quadtree-topology-worker-node.ts", import.meta.url).href;
      // Resolve through Node's require machinery so browser worker bundles do
      // not have to preserve the Node-only `import.meta.resolve` extension.
      const moduleSpecifier = "node:" + "module", urlSpecifier = "node:" + "url";
      const [{ createRequire }, { pathToFileURL }] = await Promise.all([
        import(/* @vite-ignore */ moduleSpecifier) as Promise<typeof import("node:module")>,
        import(/* @vite-ignore */ urlSpecifier) as Promise<typeof import("node:url")>
      ]);
      const tsxApiUrl = pathToFileURL(createRequire(`${process.cwd()}/package.json`).resolve("tsx/esm/api")).href;
      const source = `const { tsImport } = await import(${JSON.stringify(tsxApiUrl)}); await tsImport(${JSON.stringify(entryUrl)}, import.meta.url);`;
      const entry = new URL(`data:text/javascript,${encodeURIComponent(source)}`);
      let worker: import("node:worker_threads").Worker;
      try {
        worker = new NodeWorker(entry, { execArgv: ["--no-strip-types"] });
      } catch {
        // Node below 23.6 rejects --no-strip-types as a worker execArgv flag.
        // The data-module entry is plain JavaScript and tsx transforms the
        // real worker source, so no strip-types override is needed there.
        worker = new NodeWorker(entry);
      }
      worker.on("message", receive); worker.on("error", (error) => fail(error.message));
      cache.cpuWorker = { postMessage: (message, transfer) => worker.postMessage(message, transfer ?? []), terminate: () => worker.terminate() };
    } else return prepareQuadtreeProjectionCPU(input);
  }
  const id = (cache.cpuWorkerSequence ?? 0) + 1; cache.cpuWorkerSequence = id;
  return new Promise<PreparedProjectionCPU>((resolve, reject) => {
    cache.cpuWorkerPending!.set(id, { resolve, reject });
    const transfer: ArrayBuffer[] = [];
    if (input.packedCells.buffer instanceof ArrayBuffer) transfer.push(input.packedCells.buffer);
    if (input.columnProfiles.buffer instanceof ArrayBuffer) transfer.push(input.columnProfiles.buffer);
    if (input.opticalColumns?.buffer instanceof ArrayBuffer) transfer.push(input.opticalColumns.buffer);
    cache.cpuWorker!.postMessage({ id, input }, transfer);
  });
}
