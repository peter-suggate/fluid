/**
 * Consumer-facing ABIs for the level-set and topology publications.
 *
 * The renderer, the water pipeline and the technique overlays all read GPU
 * buffers a solver publishes. What they need is the *shape* of those
 * publications — which buffer holds what, at which stride — not the machinery
 * that fills them. Keeping the shapes here means a consumer names a
 * publication without importing the solver that produces it, which is what
 * stops the render graph from reaching the whole octree implementation.
 *
 * Nothing here allocates, binds or dispatches: these are descriptions.
 */
import type { FineLevelSetBrickPlan } from "./fine-levelset-brick-abi";

/** GPU-resident authority for one domain-global fine level-set generation. */
export interface WebGPUFineLevelSetBrickSource {
  plan: FineLevelSetBrickPlan;
  generation: number;
  generationSlot: 0 | 1;
  params: GPUBuffer;
  metadata: GPUBuffer;
  worklist: GPUBuffer;
  /** Authoritative 256-byte B4 pages: one packed u32 per sample. */
  samples: GPUBuffer;
  workA: GPUBuffer;
  workB: GPUBuffer;
  /**
   * Transactional copy of the last published signed phi. Topology rollback
   * must not alias either work buffer: transport uses workA and the Section 5
   * JFA-CPT uses both work channels for closest-point ping-pong.
   */
  rollbackSamples: GPUBuffer;
  /** GPU-published compact coarse-octree phi directory used outside fine validity. */
  coarsePhiDirectory?: GPUBuffer;
  coarsePhiRowCapacity?: number;
  /** Publication provenance; consumers use topology control only to validate coarse/fine epoch pairing. */
  topologyControl?: GPUBuffer;
  /** Diagnostic-only seed transaction control. */
  seedControl?: GPUBuffer;
}

/** Immutable sparse-cell topology plus the live CM12 field arena. Scientific
 * overlays sample these buffers directly; no finest-domain ownership or field
 * texture is materialized for visualization. */
export interface SparseAdaptiveGridConsumerSource {
  readonly kind: "sparse-adaptive-grid-sampling";
  readonly params: GPUBufferBinding;
  readonly topology: GPUBufferBinding;
  readonly state: GPUBufferBinding;
  readonly activity: GPUBufferBinding;
  readonly fineMetadata: GPUBufferBinding;
  readonly fineWorklist: GPUBufferBinding;
  readonly fineSamples: GPUBufferBinding;
}

/**
 * One compact coarse-octree phi row: `{phi, minimumPhi, maximumPhi, flags}`.
 *
 * The interval is not a volume fraction. It conservatively records the fine
 * signed-distance extrema the row was restricted from, so a consumer can tell
 * "this leaf definitely contains the interface" from "this leaf is deep
 * liquid" without the fine band being resident — and so restricting a thin
 * sheet cannot erase its zero crossing.
 */
export const OCTREE_COARSE_PHI_BYTES = 16;

/**
 * Flag bits in the fourth word of a coarse phi row.
 *
 * `valid` alone is not enough to sample: a row may be valid and non-finite
 * while its distance is still being swept, which is why `finite` is a separate
 * bit rather than an implication. Consumers that read only `valid` sampled
 * infinities into the surface for the life of the early fine-band work.
 */
export const OCTREE_COARSE_PHI_FLAG = Object.freeze({
  valid: 1 << 0,
  correctedFromFine: 1 << 1,
  containsInterface: 1 << 2,
  finite: 1 << 3,
} as const);

/** Compact background-octree phi presented without any Section-5 fine band. */
export interface CoarseLevelSetConsumerSource {
  readonly kind: "coarse-levelset-sampling";
  readonly directory: GPUBufferBinding;
  readonly control: GPUBufferBinding;
  /** Optional affine gradient `{xyz, valid}` for each directory row. */
  readonly gradients?: GPUBufferBinding;
  readonly rowCapacity: number;
  readonly sampleDimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly domainOrigin: readonly [number, number, number];
  readonly generation: number;
}

export interface OctreeTechniqueDebugSource {
  readonly leaves: GPUBufferBinding;
  readonly topologyMetrics: GPUBufferBinding;
  readonly catalogEntryHeaders: GPUBufferBinding;
  readonly catalogFaces: GPUBufferBinding;
  readonly tetrahedronHeaders: GPUBufferBinding;
  readonly tetrahedra: GPUBufferBinding;
  readonly tetrahedronVertices: GPUBufferBinding;
  readonly structuredAuthority: GPUBufferBinding;
  readonly structuredParams: GPUBufferBinding;
  readonly structuredRowGeometry: GPUBufferBinding;
  readonly structuredRowVelocities: GPUBufferBinding;
  readonly structuredControl: GPUBufferBinding;
  /**
   * Accepted Losasso adaptive leaf graph and its nodal velocity field.
   *
   * The velocity-arrow overlay deliberately reads this graph directly: each
   * 64-byte leaf record names its eight corner nodes, and bank zero of the
   * nodal arena is the accepted velocity publication. Keeping the source
   * optional makes the view fail closed while the factor-one adaptive path is
   * not active instead of falling back to the legacy structured field.
   */
  readonly losassoAdaptiveVelocity?: {
    readonly control: GPUBufferBinding;
    readonly leaves: GPUBufferBinding;
    readonly nodalVelocity: GPUBufferBinding;
    /** Current accepted per-leaf `{centre,min,max,flags}` signed distance. */
    readonly phiControl: GPUBufferBinding;
    readonly rowPhi: GPUBufferBinding;
    /** The air-side reach used by the velocity extension itself, in metres. */
    readonly extensionReach_m: number;
  };
  /** Current compact pressure potential, selected from the live ping-pong bank. */
  readonly pressure: GPUBufferBinding;
  /** Published pressure rows; diagonal and RHS stay live entirely on GPU. */
  readonly leafHeaders: GPUBufferBinding;
  /**
   * One `{phi, minimumPhi, maximumPhi, flags}` per compact row: the Section 5
   * correction as the pressure solve receives it.
   *
   * This is the single number the whole fine band exists to deliver — the free
   * surface condition of section 4.1 is built from phi at cell centres — so the
   * cell trace reads it to explain a row's diagonal rather than reporting the
   * diagonal as a bare float. Optional because a scene may run before the
   * coarse level set publishes; zero flags then mean "never corrected", which
   * is itself worth showing.
   */
  readonly coarsePhi?: GPUBufferBinding;
  /**
   * Optional live topology-rebuild publication.  The overlay expands its
   * active/retired compact tile worklist into a tiny GPU-only membership mask
   * before drawing; no list is copied or interpreted on the CPU.
   */
  readonly topologyLifecycle?: {
    readonly tileWorklist: GPUBufferBinding;
    readonly tileDimensions: readonly [number, number, number];
    readonly tileSizeCells: number;
    readonly tileCapacity: number;
  };
  /**
   * Current Section 5 sparse fine-level-set generation and its transactional
   * controls.  The overlay hashes positions into this publication exactly as
   * a consumer does, exposing missing/coarse fallback, support, interface,
   * frontier and failed/stale publication states without readback.
   */
  readonly fineBandLifecycle?: {
    readonly params: GPUBufferBinding;
    readonly metadata: GPUBufferBinding;
    readonly worklist: GPUBufferBinding;
    /** Packed persistent flags and binary16 signed-distance samples. */
    readonly samples: GPUBufferBinding;
    readonly topologyControl: GPUBufferBinding;
    readonly redistanceControl: GPUBufferBinding;
    /**
     * Resolved closest-point seed per sample, left resident by the redistance
     * commit. Reading it is what lets the flood-provenance view draw where each
     * sample's distance came from without instrumenting the flood.
     */
    readonly seeds: GPUBufferBinding;
    /**
     * The coupled authored half-widths, in finest octree cells, so the overlay
     * can draw the master pressure reach against the surface band's actual
     * derived residency. Keeping both values explicit also exposes diagnostic
     * fine-only fault injection without making it a product control.
     */
    readonly bands: {
      /** Authored pressure refinement half-width, in finest octree cells. */
      readonly pressureBandCells: number;
      /** Authored surface-tracking half-width, in finest octree cells. */
      readonly surfaceBandCells: number;
      /** Planner output: the width Section 5 transport actually moves, in fine cells. */
      readonly transportBandFineCells: number;
      /** Planner output: the width redistance leaves valid, in fine cells. */
      readonly redistanceBandFineCells: number;
      /**
       * The JFA ladder the last redistance encode emitted. The provenance view
       * bins each sample by the leading passes whose combined reach covers its
       * hop, so the colours name passes of the schedule that actually ran.
       */
      readonly ladderStrides: readonly number[];
    };
  };
  /**
   * Finest-grid shape and compact row capacity. The blast-radius cone floods the
   * level-0 row graph through the owner map, so it needs the extent of that map
   * and the size of the row field it writes.
   */
  readonly pressureRows: {
    readonly dimensions: readonly [number, number, number];
    readonly rowCapacity: number;
  };
  readonly generation: number;
}
