import type { GPUVelocityTransport } from "../../core/webgpu-eulerian";

/** Dense textures and buffers owned exclusively by the uniform reference. */
export interface UniformHostAllocationPlan {
  readonly velocityExtent: readonly [number, number, number];
  readonly transportExtent: readonly [number, number, number];
  /** Legacy extent retained for allocation-report consumers; no flux texture is allocated. */
  readonly fluxExtent: readonly [number, number, number];
  readonly pressureExtent: readonly [number, number, number];
  readonly volumeExtent: readonly [number, number, number];
  readonly conditioningBytes: number;
  readonly velocityBytes: number;
  readonly scalarBytes: number;
  /** Bytes in one scalar-packed negative x/y/z MAC boundary field. */
  readonly boundaryVelocityBytes: number;
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
  const boundaryVelocityBytes = (ny * nz + nx * nz + nx * ny) * 4;
  const velocityBytes = nx * ny * nz * velocityCopies * 16
    // Four persistent ping-pong fields own the negative x/y/z MAC boundary
    // faces, which cannot be represented by the positive-face cell packing.
    + 4 * boundaryVelocityBytes
    + (nx + 2) * (ny + 2) * (nz + 2) * transportCopies * 16
    // Sec. 3.3 owns six rgba32 scratch fields: value and Eikonal-distance
    // ping-pong pairs plus canonical resolved value/distance fields. These
    // replace, rather than supplement, the removed JFA coordinate fields.
    + (nx + 2) * (ny + 2) * (nz + 2) * 6 * 16;
  // Two pressure, two surface-density, two persistent gamma, two render-only
  // post-processing textures, and one mirrored gamma-diffusion scratch.
  // Presentation reconstruction must never double as solver scratch: sparse
  // presentation dispatches intentionally leave its exterior untouched.
  const scalarBytes = nx * ny * nz * 9 * 4;
  // Sec. 3.4 uses three fixed-point scatter fields: beta, rho deficits, and
  // gamma deficits. Sec. 3.5 reuses the first field for sharpening deposits.
  const conditioningBytes = nx * ny * nz * 3 * 4;
  return {
    velocityExtent: [nx, ny, nz],
    transportExtent: [nx + 2, ny + 2, nz + 2],
    fluxExtent: [nx, ny, nz],
    pressureExtent: [nx, ny, nz],
    volumeExtent: [nx, ny, nz],
    conditioningBytes,
    boundaryVelocityBytes,
    velocityBytes,
    scalarBytes,
    allocatedBytes: velocityBytes + scalarBytes + conditioningBytes,
  };
}
