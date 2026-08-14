/**
 * What a caller of `WebGPUOctreeProjection` has to be able to name.
 *
 * The Eulerian driver constructs the projection, reports its allocation
 * progress stage by stage, fences the t=0 bring-up one authority at a time,
 * and hands it a boundary callback that cuts the recurring frame into
 * attributable phases. All four are vocabulary, not machinery: nothing here
 * holds a device, allocates, or dispatches.
 *
 * They sit outside the projection module so a driver, a panel, or a benchmark
 * can name a stage or a phase without importing the class and every coarse
 * backend it reaches.
 */
import type { OctreeCoarseDynamicsConfiguration } from "./octree-coarse-backend";
import type { OctreeRuntimeDials } from "./octree-runtime-dials";

/** Ordered, individually fenced t=0 authority checkpoints.
 * Aanjaneya et al. (2017), Section 5 p.8, first constructs regular octree-face
 * neighborhoods, augments T-junctions with local Delaunay tetrahedra, extends
 * air velocities from physical closest points, and only then interpolates the
 * result back to power faces. */
export const OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES = [
  { id: "cold-topology", label: "Cold octree topology" },
  { id: "structured-authority", label: "Direct structured velocity and pressure authority" },
  { id: "surface-global-fine", label: "Surface and global-fine redistance publication" },
  { id: "sparse-render-world", label: "Sparse render world publication" },
] as const;
export type OctreeInitialSparseAuthorityPhaseId = typeof OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES[number]["id"];

export interface OctreeProjectionOptions {
  /** Construction-time coarse backend seam; never represented in WGSL. */
  coarseDynamics?: OctreeCoarseDynamicsConfiguration;
  maximumLeafSize?: 2 | 4 | 8 | 16 | 32;
  /** Default liquid-proximity boundary gate; false selects the unconditional control. */
  fluidGatedBoundaryRefinement?: boolean;
  /** Renderer-owned refinement of authored-environment bricks. */
  environmentBrickRefinementLevels?: number;
  /** 0 = finest cells everywhere; 1 = full distance-graded coarsening. */
  adaptivity?: number;
  /** Pure-phase cells farther from liquid/solid interfaces than this finest-cell band may remain coarse. */
  interfaceRefinementBandCells?: number;
  /**
   * Number of candidate-cell widths retained at each dyadic pressure level
   * around the surface. One preserves the sharpest legal 2:1 transition;
   * larger values create progressively wider intermediate-level shells.
   */
  surfaceRefinementGradingLayers?: number;
  /** Initial live policy for the fenced t=0 topology; authored fields still size allocations. */
  initialRuntimeDials?: OctreeRuntimeDials;
  /**
   * Half-width of the Section 5 high-resolution surface-tracking band, in
   * finest octree cells. The product master control supplies the same authored
   * reach as the pressure band above; the separate option remains only for
   * diagnostic fault injection. An unset value follows
   * `interfaceRefinementBandCells`.
   */
  fineLevelSetBandCells?: number;
  /** Authoritative domain-global Section 5 narrow-band factor. */
  globalFineLevelSetFactor?: 1 | 4 | 8;
  /** Explicit physical brick cap for the global factor-1/factor-4/factor-8 publication. */
  globalFineLevelSetMaximumBricks?: number;
  /** Advanced safety override for the compact pressure-row arena. */
  pressureRowCapacity?: number;
}

/** Allocation milestones owned by the octree resource graph. The constructor
 * that performs the work reports against this list stage by stage, so a new
 * GPU boundary that does not appear here is progress the product never sees. */
export const OCTREE_ALLOCATION_STAGES = Object.freeze([
  "Plan octree domain and capacity",
  "Allocate sparse brick-world resources",
  "Allocate topology residency scheduler",
  "Allocate analytic bootstrap and surface state",
  "Allocate topology owners and pressure rows",
  "Build octree layouts and bind groups",
  "Allocate fine-interface seed resources",
  "Allocate global fine level-set pages",
  "Finalize octree resource graph",
] as const);

export type OctreeAllocationProgress = (label: string, completed: number, total: number) => void;

/** The seven data-domain boundaries of the collapsed recurring frame. */
const OCTREE_ENGINE_PHASES = [
  "structureEpoch",
  "rowEngineA",
  "solveEngine",
  "rowEngineB",
  "brickEngineA",
  "closestPointWaves",
  "brickEngineB",
] as const;
export type OctreeEnginePhase = typeof OCTREE_ENGINE_PHASES[number];

/** Fine-grained checkpoints for direct structured attribution. */
export const OCTREE_FINE_SEMANTIC_PHASES = [
  "powerDescriptorTopology",
  "structuredAdvectionBoundaryRhs",
  "structuredVolumeCapture",
  "finalPressureRowAssembly",
  "mgpcgSolve",
  "structuredProjection",
  "structuredProjectionTail",
  "finePreparation",
  "fineTransport",
  "fineTopology",
  "fineRedistance",
  "fineRestriction",
] as const;
export type OctreeFineSemanticPhase = typeof OCTREE_FINE_SEMANTIC_PHASES[number];
/** Semantic checkpoints consumed by the generic adjacent-boundary trace.
 * Instrumented runs use numerical seams by default. Set
 * `FLUID_ENGINE_SPLIT=collapsed` only for the seven-bucket engine view. */
export type OctreeSemanticPhase = OctreeEnginePhase | OctreeFineSemanticPhase;
export type OctreeSemanticBoundary = (
  phase: OctreeSemanticPhase,
  encoder: GPUCommandEncoder,
) => GPUCommandEncoder;

/** The host-owned buffers every octree projection binds, whatever its lane. */
export interface OctreeProjectionResources {
  rigidBodies: GPUBuffer;
  rigidExchange: GPUBuffer;
  rigidImmersedVolumes: GPUBuffer;
  terrain: GPUTexture;
}
