import type { GPUVelocityTransport } from "../../core/webgpu-eulerian";
import type { UniformLatticeAddress } from "./uniform-lattice-address";

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
  address: UniformLatticeAddress,
  transport: GPUVelocityTransport,
): UniformHostAllocationPlan {
  const [nx, ny, nz] = address.cells, cells = address.boxCells;
  const velocityCopies = transport === "maccormack" ? 4 : 2;
  const transportCopies = transport === "maccormack" ? 2 : 1;
  const boundaryVelocityBytes = address.lowFaceKeys * 4;
  const velocityBytes = cells * velocityCopies * 16
    // Four persistent ping-pong fields own the negative x/y/z MAC boundary
    // faces, which cannot be represented by the positive-face cell packing.
    + 4 * boundaryVelocityBytes
    + (nx + 2) * (ny + 2) * (nz + 2) * transportCopies * 16;
  // Two pressure, two surface-density, two persistent gamma, two render-only
  // post-processing textures, and one mirrored gamma-diffusion scratch.
  // Presentation reconstruction must never double as solver scratch: sparse
  // presentation dispatches intentionally leave its exterior untouched.
  const scalarBytes = cells * 9 * 4;
  // Sec. 3.4 uses three fixed-point scatter fields: beta, rho deficits, and
  // gamma deficits. Sec. 3.5 reuses the first field for sharpening deposits.
  const conditioningBytes = cells * 3 * 4;
  return {
    velocityExtent: address.cells,
    transportExtent: [nx + 2, ny + 2, nz + 2],
    fluxExtent: address.cells,
    pressureExtent: address.cells,
    volumeExtent: address.cells,
    conditioningBytes,
    boundaryVelocityBytes,
    velocityBytes,
    scalarBytes,
    allocatedBytes: velocityBytes + scalarBytes + conditioningBytes,
  };
}
