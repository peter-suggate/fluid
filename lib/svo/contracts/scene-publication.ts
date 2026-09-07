/** Accepted scene payload shared by construction and renderer hosts. */
import type { SvoPrimitiveCandidatePublication } from "../features/scene-publication/svo-primitive-candidates";

export interface SparseVoxelDrySceneData {
  /** Monotonic renderer publication, independent of the solver generation. */
  renderRevision: number;
  /** Packed `SvoPrimitiveRecord` values in dense live-scene publication order. */
  primitiveRecords: Uint32Array<ArrayBuffer>;
  /** Required exact secondary-ray acceleration for the same primitive records. */
  primitiveCandidates: SvoPrimitiveCandidatePublication;
  /** Complete live material table. Binding 6 is renderer-owned and capacity-stable. */
  materialRecords: Uint32Array<ArrayBuffer>;
  materialRevision: number;
  /** First owner ID belonging to primitive zero (rigid bodies occupy the ids below it). */
  ownerBase: number;
  /** Interior-facing shell pane omitted so the camera can see into the room. */
  skippedOwnerId?: number;
  /**
   * Packed aggregate parameter blocks, in publication order, as
   * `packSvoDrySceneClusters` produces them. Absent means this scene publishes
   * no aggregates, and the region is uploaded as zeroes rather than left alone.
   */
  clusterBlocks?: Uint32Array<ArrayBuffer>;
  /**
   * Packed field-program tape blocks, in publication order, as
   * `packSvoDrySceneFieldPrograms` produces them. Absent means this scene
   * publishes no tapes, and the region is uploaded as zeroes rather than left
   * alone — a stale block from the previous scene would otherwise be resolved by
   * a new scene's record and grow somebody else's shape.
   */
  fieldProgramBlocks?: Uint32Array<ArrayBuffer>;
  /** Packed 80-byte finite-pane records. Empty means this scene has no glass. */
  glassRecords?: Uint32Array<ArrayBuffer>;
  /** Versioned live content key used to avoid redundant pane uploads. */
  glassCacheKey?: string;
  /** Packed analytic sphere/ellipsoid glass records mirrored into a renderer-owned uniform arena. */
  thickGlassRecords?: Uint32Array<ArrayBuffer>;
  thickGlassRevision?: number;
  thickGlassCacheKey?: string;
  /** Thin pane replaced by a curved volume only while the thick binder is valid. */
  thickGlassReplacedThinPaneId?: number;
  /** CPU-built mirror of the producer's bounded 112-byte light publication. */
  lightRecords?: Uint32Array<ArrayBuffer>;
  /** CPU-built mirror revision; must equal the authoritative source publication. */
  lightRevision?: number;
  /** CPU-built mirror of the selected 96-byte environment-lighting record. */
  environmentLightingRecord?: Uint32Array<ArrayBuffer>;
  /** Content identity; must equal the authoritative source publication. */
  environmentLightingCacheKey?: string;
  /** Scene capability gate for bounded indirect-diffuse contact visibility. */
  contactVisibilityEnabled?: boolean;
  /** Scene capability gate for shadow visibility; omission keeps shadows available. */
  shadowVisibilityEnabled?: boolean;
  /** Use the entered voxel face instead of sub-voxel tangent-surface reconstruction. */
  flatVoxelNormals?: boolean;
  lightDirection?: readonly [number, number, number];
  lightColor?: readonly [number, number, number];
}
