import type { GPUVelocityTransport } from "./webgpu-eulerian";

export interface OctreeHostAllocationPlan {
  readonly cutover: boolean;
  readonly velocityExtent: readonly [number, number, number];
  readonly transportExtent: readonly [number, number, number];
  readonly fluxExtent: readonly [number, number, number];
  readonly pressureExtent: readonly [number, number, number];
  readonly volumeExtent: readonly [number, number, number];
  readonly conditioningBytes: number;
  readonly denseConditioningBaselineBytes: number;
  /** Bytes actually allocated for velocity, transport, and flux textures. */
  readonly velocityAllocatedBytes: number;
  /** Bytes actually allocated for pressure and volume textures. */
  readonly scalarAllocatedBytes: number;
  /** Bytes actually allocated for all box-sized host compatibility fields. */
  readonly allocatedBytes: number;
  readonly denseVelocityBaselineBytes: number;
  readonly denseScalarBaselineBytes: number;
  /** Bytes all equivalent dense host textures would allocate. */
  readonly denseBaselineBytes: number;
  readonly velocitySavedBytes: number;
  readonly scalarSavedBytes: number;
  readonly savedBytes: number;
}

/** @deprecated Use OctreeHostAllocationPlan. */
export type OctreeHostVelocityAllocationPlan = OctreeHostAllocationPlan;

/**
 * Exact host-side allocation contract for the U3 compact-face cutover.
 *
 * WebGPU bind-group layouts remain shared with the uniform solver, so the
 * cutover keeps format-correct 1x1x1 textures for bindings that cannot yet be
 * removed. They are never dispatched over the simulation domain. The compact
 * face buffers themselves are accounted by WebGPUOctreeProjection.
 */
export function planOctreeHostAllocation(
  nx: number,
  ny: number,
  nz: number,
  transport: GPUVelocityTransport,
  cutover: boolean,
): OctreeHostAllocationPlan {
  for (const [name, value] of Object.entries({ nx, ny, nz })) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`);
  }
  const velocityCopies = transport === "maccormack" ? 4 : 2;
  const transportCopies = transport === "maccormack" ? 2 : 1;
  const denseVelocityBytes = nx * ny * nz * velocityCopies * 16;
  const denseTransportBytes = (nx + 2) * (ny + 2) * (nz + 2) * transportCopies * 8;
  const denseFluxBytes = nx * ny * nz * 8;
  const denseVelocityBaselineBytes = denseVelocityBytes + denseTransportBytes + denseFluxBytes;
  // Two r32float pressure fields and two r32float volume fields.
  const denseScalarBaselineBytes = nx * ny * nz * 4 * 4;
  // The conservative-volume sharpening scatter arena is dormant for octree.
  const denseConditioningBaselineBytes = nx * ny * nz * 4;
  const denseBaselineBytes = denseVelocityBaselineBytes + denseScalarBaselineBytes + denseConditioningBaselineBytes;
  const velocityCompatibilityBytes = velocityCopies * 16 + transportCopies * 8 + 8;
  const scalarCompatibilityBytes = 4 * 4;
  const velocityAllocatedBytes = cutover ? velocityCompatibilityBytes : denseVelocityBaselineBytes;
  const scalarAllocatedBytes = cutover ? scalarCompatibilityBytes : denseScalarBaselineBytes;
  const conditioningBytes = cutover ? 4 : denseConditioningBaselineBytes;
  return {
    cutover,
    velocityExtent: cutover ? [1, 1, 1] : [nx, ny, nz],
    transportExtent: cutover ? [1, 1, 1] : [nx + 2, ny + 2, nz + 2],
    fluxExtent: cutover ? [1, 1, 1] : [nx, ny, nz],
    pressureExtent: cutover ? [1, 1, 1] : [nx, ny, nz],
    volumeExtent: cutover ? [1, 1, 1] : [nx, ny, nz],
    conditioningBytes,
    denseConditioningBaselineBytes,
    velocityAllocatedBytes,
    scalarAllocatedBytes,
    allocatedBytes: velocityAllocatedBytes + scalarAllocatedBytes + conditioningBytes,
    denseVelocityBaselineBytes,
    denseScalarBaselineBytes,
    denseBaselineBytes,
    velocitySavedBytes: cutover ? denseVelocityBaselineBytes - velocityCompatibilityBytes : 0,
    scalarSavedBytes: cutover ? denseScalarBaselineBytes - scalarCompatibilityBytes : 0,
    savedBytes: cutover ? denseBaselineBytes - velocityCompatibilityBytes - scalarCompatibilityBytes - conditioningBytes : 0,
  };
}

/** @deprecated Use planOctreeHostAllocation. */
export const planOctreeHostVelocityAllocation = planOctreeHostAllocation;
