/** Reduced axis-face velocity sampling ABI shared by Losasso fine consumers. */

export const OCTREE_LOSASSO_AXIS_FACE_GEOMETRY_WORDS = 4;
export const OCTREE_LOSASSO_AXIS_FACE_DIRECTORY_WORDS = 2;
export const OCTREE_LOSASSO_AXIS_FACE_MAXIMUM_PROBES = 32;

export interface WebGPUOctreeLosassoVelocitySamplerSource {
  /** Accepted coarse authority `[epoch,rowCount,faceCount,valid,...]`. */
  readonly control: GPUBuffer;
  /** `vec4u(axis | (log2(span)<<2),x,y,z)` in finest velocity-grid coordinates. */
  readonly faceGeometry: GPUBuffer;
  /** One f32 normal velocity per face, after fixed-K extension. */
  readonly extendedVelocity: GPUBuffer;
  /** Open-addressed `vec2u(facePlusOne,hash)` records. Zero is empty. */
  readonly axisFaceDirectory: GPUBuffer;
  /** Dense finest-MAC lattice, finite value bits or a reserved invalid NaN. */
  readonly stagedVelocity?: GPUBuffer;
  readonly directoryCapacity: number;
  /** Finest-cell dimensions; the selected axis admits coordinate `dimension`. */
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  /** Finest velocity-grid spacing Δx, not the factor-4/8 phi spacing. */
  readonly fineCellSize: number;
}

export interface OctreeLosassoAxisFaceDirectoryPlan {
  readonly faceCapacity: number;
  readonly directoryCapacity: number;
  readonly loadFactor: number;
  readonly allocatedBytes: number;
}

export function planOctreeLosassoAxisFaceDirectory(
  faceCapacity: number,
): OctreeLosassoAxisFaceDirectoryPlan {
  if (!Number.isSafeInteger(faceCapacity) || faceCapacity < 1) {
    throw new RangeError("Losasso sampler face capacity must be positive");
  }
  let directoryCapacity = 1;
  // The shared publisher/sampler ABI deliberately bounds linear probing to
  // 32 slots. Keep the table at or below one-quarter load so a valid compact
  // face set does not fail publication merely from a long primary cluster.
  while (directoryCapacity < 4 * faceCapacity) directoryCapacity *= 2;
  if (directoryCapacity > 0x4000_0000) throw new RangeError("Losasso face directory exceeds u32 addressing");
  return Object.freeze({ faceCapacity, directoryCapacity,
    loadFactor: faceCapacity / directoryCapacity,
    allocatedBytes: directoryCapacity * OCTREE_LOSASSO_AXIS_FACE_DIRECTORY_WORDS * 4 });
}

export function octreeLosassoAxisFaceHash(
  packedAxisSpan: number, x: number, y: number, z: number,
): number {
  let value = Math.imul((packedAxisSpan + 1) >>> 0, 0x9e37_79b1) >>> 0;
  value = Math.imul((value ^ (x >>> 0)) >>> 0, 0x85eb_ca6b) >>> 0;
  value = Math.imul((value ^ (y >>> 0)) >>> 0, 0xc2b2_ae35) >>> 0;
  return Math.imul((value ^ (z >>> 0)) >>> 0, 0x27d4_eb2d) >>> 0;
}

/** Build the immutable lookup image installed alongside a face publication. */
export function buildOctreeLosassoAxisFaceDirectory(
  faceGeometry: Uint32Array,
  dimensions: readonly [number, number, number],
  directoryCapacity = planOctreeLosassoAxisFaceDirectory(
    faceGeometry.length / OCTREE_LOSASSO_AXIS_FACE_GEOMETRY_WORDS).directoryCapacity,
): Uint32Array {
  if (faceGeometry.length === 0
    || faceGeometry.length % OCTREE_LOSASSO_AXIS_FACE_GEOMETRY_WORDS !== 0
    || !Number.isSafeInteger(directoryCapacity) || directoryCapacity < 2
    || (directoryCapacity & (directoryCapacity - 1)) !== 0) {
    throw new RangeError("Losasso face geometry/directory shape is invalid");
  }
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Losasso sampler dimensions must be positive integers");
  }
  const faceCount = faceGeometry.length / OCTREE_LOSASSO_AXIS_FACE_GEOMETRY_WORDS;
  if (faceCount * 4 > directoryCapacity) throw new RangeError("Losasso face directory load exceeds 0.25");
  const result = new Uint32Array(directoryCapacity * OCTREE_LOSASSO_AXIS_FACE_DIRECTORY_WORDS);
  const mask = directoryCapacity - 1;
  for (let face = 0; face < faceCount; face += 1) {
    const at = face * OCTREE_LOSASSO_AXIS_FACE_GEOMETRY_WORDS;
    const packedAxisSpan = faceGeometry[at]!, axis = packedAxisSpan & 3;
    const logSpan = packedAxisSpan >>> 2, span = 2 ** logSpan, x = faceGeometry[at + 1]!;
    const y = faceGeometry[at + 2]!, z = faceGeometry[at + 3]!;
    if (axis > 2 || !Number.isSafeInteger(span) || span < 1
      || x > dimensions[0] || y > dimensions[1] || z > dimensions[2]
      || (axis !== 0 && x === dimensions[0])
      || (axis !== 1 && y === dimensions[1])
      || (axis !== 2 && z === dimensions[2])
      || (axis !== 0 && x % span !== 0) || (axis !== 1 && y % span !== 0)
      || (axis !== 2 && z % span !== 0)) {
      throw new RangeError(`Losasso face ${face} is outside its staggered lattice`);
    }
    const hash = octreeLosassoAxisFaceHash(packedAxisSpan, x, y, z);
    let installed = false;
    for (let probe = 0; probe < OCTREE_LOSASSO_AXIS_FACE_MAXIMUM_PROBES; probe += 1) {
      const slot = ((hash + probe) & mask) * OCTREE_LOSASSO_AXIS_FACE_DIRECTORY_WORDS;
      if (result[slot] === 0) {
        result[slot] = face + 1; result[slot + 1] = hash; installed = true; break;
      }
    }
    if (!installed) throw new RangeError("Losasso face directory exceeded its bounded probe window");
  }
  return result;
}
