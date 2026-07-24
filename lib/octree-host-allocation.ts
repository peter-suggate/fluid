import type { GPUVelocityTransport } from "./webgpu-eulerian";

/**
 * Dense host fields retained by the uniform and quadtree methods.
 *
 * The Power-liquid octree does not have a mode in this plan: its compact
 * velocity, pressure, and level-set owners allocate themselves.
 */
export interface UniformHostAllocationPlan {
  readonly velocityExtent: readonly [number, number, number];
  readonly transportExtent: readonly [number, number, number];
  readonly fluxExtent: readonly [number, number, number];
  readonly pressureExtent: readonly [number, number, number];
  readonly volumeExtent: readonly [number, number, number];
  readonly conditioningBytes: number;
  readonly velocityBytes: number;
  readonly scalarBytes: number;
  readonly allocatedBytes: number;
}

export function planUniformHostAllocation(
  nx: number,
  ny: number,
  nz: number,
  transport: GPUVelocityTransport,
): UniformHostAllocationPlan {
  for (const [name, value] of Object.entries({ nx, ny, nz })) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive safe integer`);
    }
  }
  const velocityCopies = transport === "maccormack" ? 4 : 2;
  const transportCopies = transport === "maccormack" ? 2 : 1;
  const velocityBytes = nx * ny * nz * velocityCopies * 16
    + (nx + 2) * (ny + 2) * (nz + 2) * transportCopies * 8
    + nx * ny * nz * 8;
  const scalarBytes = nx * ny * nz * 4 * 4;
  const conditioningBytes = nx * ny * nz * 4;
  return {
    velocityExtent: [nx, ny, nz],
    transportExtent: [nx + 2, ny + 2, nz + 2],
    fluxExtent: [nx, ny, nz],
    pressureExtent: [nx, ny, nz],
    volumeExtent: [nx, ny, nz],
    conditioningBytes,
    velocityBytes,
    scalarBytes,
    allocatedBytes: velocityBytes + scalarBytes + conditioningBytes,
  };
}
