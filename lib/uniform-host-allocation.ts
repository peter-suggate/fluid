import type { GPUVelocityTransport } from "./webgpu-eulerian";

/** Dense textures and buffers owned exclusively by the uniform reference. */
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
  // Two pressure, two conservative VOF, and two render-only smoothed surface
  // textures. Presentation smoothing must never feed back into transport.
  const scalarBytes = nx * ny * nz * 6 * 4;
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
