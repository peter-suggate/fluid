/** Measured pressure/MG page shape and its offline CPU numerical oracle.
 *
 * The live first-order degree-2/4 page smoother is
 * `smoothPageChebyshevForward/Reverse` in the SPGrid V-cycle. Section 6.3
 * coefficient channels are published into that hierarchy and the accurate A2
 * operator; they do not select another page executor. This module therefore
 * contains no GPU class, WGSL, dispatch record, or alternate pressure path.
 */

const OCTREE_PRESSURE_MG_PAGE_SHAPE = Object.freeze([8, 8, 4] as const);
const OCTREE_PRESSURE_MG_PAGE_WORKGROUP_SIZE = 128 as const;
const OCTREE_PRESSURE_MG_PAGE_ELEMENTS = 256 as const;
const OCTREE_PRESSURE_MG_HALO_ELEMENTS = 600 as const;
const OCTREE_PRESSURE_MG_HALO_AMPLIFICATION = 600 / 256;
const OCTREE_PRESSURE_MG_LOCAL_F32_CHANNELS = 4 as const;
const OCTREE_PRESSURE_MG_LOCAL_INDEX_CHANNELS = 1 as const;
const OCTREE_PRESSURE_MG_WORKGROUP_BYTES = 12_000 as const;

export interface OctreePressureMGPageChebyshevPlan {
  readonly pageShape: typeof OCTREE_PRESSURE_MG_PAGE_SHAPE;
  readonly haloShape: readonly [10, 10, 6];
  readonly pageElements: 256;
  readonly haloElements: 600;
  readonly workgroupSize: 128;
  readonly localF32Channels: 4;
  readonly localIndexChannels: 1;
  readonly workgroupBytes: number;
  readonly haloAmplification: number;
  readonly localChebyshevSweeps: 4;
}

export function planOctreePressureMGPageChebyshev(): OctreePressureMGPageChebyshevPlan {
  return Object.freeze({
    pageShape: OCTREE_PRESSURE_MG_PAGE_SHAPE,
    haloShape: [10, 10, 6] as const,
    pageElements: OCTREE_PRESSURE_MG_PAGE_ELEMENTS,
    haloElements: OCTREE_PRESSURE_MG_HALO_ELEMENTS,
    workgroupSize: OCTREE_PRESSURE_MG_PAGE_WORKGROUP_SIZE,
    localF32Channels: OCTREE_PRESSURE_MG_LOCAL_F32_CHANNELS,
    localIndexChannels: OCTREE_PRESSURE_MG_LOCAL_INDEX_CHANNELS,
    workgroupBytes: OCTREE_PRESSURE_MG_WORKGROUP_BYTES,
    haloAmplification: OCTREE_PRESSURE_MG_HALO_AMPLIFICATION,
    localChebyshevSweeps: 4,
  });
}

export function octreePressureMGInteriorHaloIndex(
  local: readonly [number, number, number],
): number {
  if (local.some((value, axis) => !Number.isSafeInteger(value)
    || value < 0 || value >= OCTREE_PRESSURE_MG_PAGE_SHAPE[axis])) {
    throw new RangeError("Pressure/MG local coordinate is outside 8x8x4");
  }
  return local[0] + 1 + 10 * (local[1] + 1 + 10 * (local[2] + 1));
}

/** CPU numerical oracle for the exact four local page sweeps. */
export function applyOctreePressureMGPageChebyshev(
  inputHalo: Float32Array,
  rhsHalo: Float32Array,
  diagonalHalo: Float32Array,
  weights: readonly [number, number, number, number],
): Float32Array {
  if ([inputHalo, rhsHalo, diagonalHalo]
    .some((field) => field.length !== OCTREE_PRESSURE_MG_HALO_ELEMENTS)) {
    throw new RangeError("Pressure/MG page oracle requires 600-value halo fields");
  }
  let source = Float32Array.from(inputHalo);
  let target = Float32Array.from(inputHalo);
  for (const weight of weights) {
    for (let z = 0; z < 4; z += 1) for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const index = octreePressureMGInteriorHaloIndex([x, y, z]);
        const diagonal = diagonalHalo[index]!;
        if (!(diagonal > 0) || !Number.isFinite(diagonal)) {
          throw new RangeError("Pressure/MG page oracle requires positive finite diagonals");
        }
        const applied = diagonal * source[index]!
          - source[index - 1]! - source[index + 1]!
          - source[index - 10]! - source[index + 10]!
          - source[index - 100]! - source[index + 100]!;
        target[index] = Math.fround(source[index]!
          + weight * (rhsHalo[index]! - applied) / diagonal);
      }
    }
    [source, target] = [target, source];
  }
  const output = new Float32Array(OCTREE_PRESSURE_MG_PAGE_ELEMENTS);
  let cursor = 0;
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      output[cursor] = source[octreePressureMGInteriorHaloIndex([x, y, z])]!;
      cursor += 1;
    }
  }
  return output;
}
