/**
 * Read-only GPU publications consumed by the paper-technique overlay.
 *
 * This deliberately exposes existing compact buffers rather than creating a
 * dense diagnostic field or reading topology back to the CPU.  The draw
 * pipelines use disjoint subsets of the bundle so each remains below WebGPU's
 * portable fragment-stage storage-buffer limit.
 */
export interface OctreeTechniqueDebugSource {
  readonly leaves: GPUBufferBinding;
  readonly topologyMetrics: GPUBufferBinding;
  readonly tetrahedronHeaders: GPUBufferBinding;
  readonly tetrahedra: GPUBufferBinding;
  readonly tetrahedronVertices: GPUBufferBinding;
  readonly powerFaces: GPUBufferBinding;
  readonly faceNormals: GPUBufferBinding;
  readonly faceCentroids: GPUBufferBinding;
  readonly incidenceRows: GPUBufferBinding;
  readonly incidence: GPUBufferBinding;
  readonly faceControl: GPUBufferBinding;
  /** Published pressure rows; diagonal and RHS stay live entirely on GPU. */
  readonly leafHeaders: GPUBufferBinding;
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
    readonly hash: GPUBufferBinding;
    readonly metadata: GPUBufferBinding;
    readonly worklist: GPUBufferBinding;
    readonly sampleFlags: GPUBufferBinding;
    /** Paper Section 5 signed-distance samples on the factor-m fine lattice. */
    readonly phi: GPUBufferBinding;
    readonly topologyControl: GPUBufferBinding;
    readonly redistanceControl: GPUBufferBinding;
  };
  /** Paper Section 5 regular-face march, exposed from the already-live band
   * graph. The dedicated overlay reads these buffers directly and remains
   * below the portable storage-binding limit. */
  readonly section5FaceBand?: {
    readonly rowHash: GPUBufferBinding;
    readonly rows: GPUBufferBinding;
    readonly faces: GPUBufferBinding;
    readonly incidence: GPUBufferBinding;
    readonly states: GPUBufferBinding;
    readonly control: GPUBufferBinding;
    readonly transitionControl: GPUBufferBinding;
  };
  readonly generation: number;
}

export const OCTREE_TECHNIQUE_OVERLAY_MODES = [
  "power-cells",
  "power-faces",
  "delaunay-tetrahedra",
  "transition-band",
  "power-operator",
  "octree-lifecycle",
  "fine-band-lifecycle",
  "operator-diagonal",
  "operator-rhs",
  "operator-reciprocity",
  "operator-open-fraction",
  "tetra-validity",
  "section5-face-band",
  "global-fine-phi",
] as const;

export type OctreeTechniqueOverlayMode = typeof OCTREE_TECHNIQUE_OVERLAY_MODES[number];

export function isOctreeTechniqueOverlayMode(value: string): value is OctreeTechniqueOverlayMode {
  return (OCTREE_TECHNIQUE_OVERLAY_MODES as readonly string[]).includes(value);
}

export const OCTREE_TECHNIQUE_OVERLAY_CODES: Readonly<Record<OctreeTechniqueOverlayMode, number>> = {
  "power-cells": 12,
  "power-faces": 13,
  "delaunay-tetrahedra": 14,
  "transition-band": 15,
  "power-operator": 16,
  "octree-lifecycle": 17,
  "fine-band-lifecycle": 18,
  "operator-diagonal": 19,
  "operator-rhs": 20,
  "operator-reciprocity": 21,
  "operator-open-fraction": 22,
  "tetra-validity": 23,
  "section5-face-band": 24,
  "global-fine-phi": 25,
};
