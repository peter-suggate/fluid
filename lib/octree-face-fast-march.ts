/** Deterministic CPU oracle for the regular-graph subset of Section 5 face extrapolation. */


export const OCTREE_FACE_MARCH_INVALID = 0xffff_ffff;
/** Four 2:1 subfaces on each positive axis. */
export const OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW = 12;
/** Six sides times four 2:1 subfaces is the exact regular-face incidence bound. */
export const OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW = 24;
export const OCTREE_REGULAR_BAND_FACE_BYTES = 32;

export interface OctreeFaceBandPhiSummary {
  readonly representativePhi: number;
  readonly minimumPhi: number;
  readonly maximumPhi: number;
}

/** Closest-to-interface representative without losing a mixed-sign interval. */
export function summarizeOctreeFaceBandPhi(values: readonly number[]): OctreeFaceBandPhiSummary {
  if (values.length === 0) throw new RangeError("Face-band phi summary needs at least one sample");
  let representativePhi = Number.POSITIVE_INFINITY, minimumPhi = Number.POSITIVE_INFINITY;
  let maximumPhi = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) throw new RangeError("Face-band phi samples must be finite");
    minimumPhi = Math.min(minimumPhi, value); maximumPhi = Math.max(maximumPhi, value);
    if (Math.abs(value) < Math.abs(representativePhi)
      || (Math.abs(value) === Math.abs(representativePhi) && value < representativePhi)) representativePhi = value;
  }
  return { representativePhi, minimumPhi, maximumPhi };
}

export interface OctreeFaceMarchFace {
  readonly negativeRow: number;
  readonly positiveRow: number;
  /** Signed-distance sample at the regular octree face, in metres. */
  readonly phi: number;
  /** Present only on power-to-regular interpolation seeds. */
  readonly normalVelocity?: number;
  /** False excludes a capacity face from the physical extrapolation band. */
  readonly inBand?: boolean;
}

export interface OctreeFaceMarchResult {
  readonly velocities: Float32Array;
  readonly parents: Uint32Array;
  readonly graphDistance: Uint32Array;
  readonly acceptedCount: number;
  readonly unresolvedFaces: Uint32Array;
  /** Exact number of parallel one-ring propagation rounds required. */
  readonly maximumGraphDistance: number;
}

export interface OctreeRegularFaceBandPlan {
  readonly wetRowCapacity: number;
  readonly maximumFineBricks: number;
  readonly ownerCandidatesPerBrick: number;
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly faceBytes: number;
  readonly incidenceBytes: number;
  readonly phiBytes: number;
  readonly marchBytes: number;
  readonly allocatedBytes: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

/**
 * Every active fine brick intersects at most ceil(B / m)^3 finest octree
 * cells. Deduplicating their containing owner keys can only reduce the number
 * of dry band rows. Wet pressure rows remain a disjoint, stable prefix.
 */
export function planOctreeRegularFaceBand(
  wetRowCapacityValue: number,
  maximumFineBricksValue: number,
  brickResolutionValue: number,
  fineFactorValue: number,
): OctreeRegularFaceBandPlan {
  const wetRowCapacity = positiveInteger(wetRowCapacityValue, "Wet row capacity");
  const maximumFineBricks = positiveInteger(maximumFineBricksValue, "Fine brick capacity");
  const brickResolution = positiveInteger(brickResolutionValue, "Fine brick resolution");
  const fineFactor = positiveInteger(fineFactorValue, "Fine factor");
  const ownerCandidatesPerAxis = Math.ceil(brickResolution / fineFactor);
  const ownerCandidatesPerBrick = ownerCandidatesPerAxis ** 3;
  const rowCapacity = wetRowCapacity + maximumFineBricks * ownerCandidatesPerBrick;
  if (!Number.isSafeInteger(rowCapacity)) throw new RangeError("Regular face-band row capacity exceeds exact integer range");
  const faceCapacity = rowCapacity * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW;
  const faceBytes = faceCapacity * OCTREE_REGULAR_BAND_FACE_BYTES;
  const incidenceBytes = rowCapacity * (1 + OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW) * 4;
  const phiBytes = faceCapacity * 4;
  // Two ping-pong velocity channels, two state channels, parent, and graph depth.
  const marchBytes = faceCapacity * 6 * 4;
  return { wetRowCapacity, maximumFineBricks, ownerCandidatesPerBrick, rowCapacity, faceCapacity,
    faceBytes, incidenceBytes, phiBytes, marchBytes,
    allocatedBytes: faceBytes + incidenceBytes + phiBytes + marchBytes };
}

/**
 * Oracle for Section 5's closest-face propagation rule on a supplied regular
 * face graph: process faces in increasing |phi| and copy the velocity of the
 * already-known incident face closest to the free surface. This does not model
 * the paper's Delaunay transition connectivity. Equal-distance ties use face
 * ID. The recorded graph distance is a data-derived dispatch bound for a
 * parallel one-ring GPU implementation.
 */
export function fastMarchOctreeFaceVelocity(
  rowCountValue: number,
  faces: readonly OctreeFaceMarchFace[],
  tolerance = 1e-6,
): OctreeFaceMarchResult {
  const rowCount = positiveInteger(rowCountValue, "Face-march row count");
  if (!Number.isFinite(tolerance) || tolerance < 0) throw new RangeError("Face-march tolerance must be finite and non-negative");
  const incidence = Array.from({ length: rowCount }, () => [] as number[]);
  const inBand = new Uint8Array(faces.length);
  const known = new Uint8Array(faces.length);
  const velocities = new Float32Array(faces.length);
  const parents = new Uint32Array(faces.length); parents.fill(OCTREE_FACE_MARCH_INVALID);
  const graphDistance = new Uint32Array(faces.length); graphDistance.fill(OCTREE_FACE_MARCH_INVALID);
  faces.forEach((face, index) => {
    if (!Number.isFinite(face.phi)) throw new RangeError(`Face ${index} phi must be finite`);
    for (const row of [face.negativeRow, face.positiveRow]) {
      if (row === OCTREE_FACE_MARCH_INVALID) continue;
      if (!Number.isSafeInteger(row) || row < 0 || row >= rowCount) throw new RangeError(`Face ${index} row is outside the band`);
      incidence[row].push(index);
    }
    inBand[index] = face.inBand === false ? 0 : 1;
    if (face.normalVelocity !== undefined) {
      if (!Number.isFinite(face.normalVelocity)) throw new RangeError(`Face ${index} seed velocity must be finite`);
      if (inBand[index] !== 0) {
        known[index] = 1; velocities[index] = face.normalVelocity; parents[index] = index; graphDistance[index] = 0;
      }
    }
  });
  incidence.forEach((row, index) => {
    row.sort((a, b) => a - b);
    if (row.length > OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW) {
      throw new RangeError(`Face-march row ${index} exceeds the proved incidence bound`);
    }
  });
  const order = faces.map((_, index) => index).filter((index) => inBand[index] !== 0)
    .sort((a, b) => Math.abs(faces[a].phi) - Math.abs(faces[b].phi) || a - b);
  for (const faceIndex of order) {
    if (known[faceIndex] !== 0) continue;
    const face = faces[faceIndex];
    const candidates = new Set<number>();
    for (const row of [face.negativeRow, face.positiveRow]) {
      if (row === OCTREE_FACE_MARCH_INVALID) continue;
      incidence[row].forEach((candidate) => candidates.add(candidate));
    }
    let source = OCTREE_FACE_MARCH_INVALID;
    for (const candidate of candidates) {
      if (known[candidate] === 0 || candidate === faceIndex) continue;
      if (Math.abs(faces[candidate].phi) > Math.abs(face.phi) + tolerance) continue;
      if (source === OCTREE_FACE_MARCH_INVALID
        || Math.abs(faces[candidate].phi) < Math.abs(faces[source].phi)
        || (Math.abs(faces[candidate].phi) === Math.abs(faces[source].phi) && candidate < source)) source = candidate;
    }
    if (source !== OCTREE_FACE_MARCH_INVALID) {
      known[faceIndex] = 1; velocities[faceIndex] = velocities[source]; parents[faceIndex] = source;
      graphDistance[faceIndex] = graphDistance[source] + 1;
    }
  }
  const unresolved = order.filter((index) => known[index] === 0);
  let maximumGraphDistance = 0;
  for (const index of order) if (known[index] !== 0) maximumGraphDistance = Math.max(maximumGraphDistance, graphDistance[index]);
  return { velocities, parents, graphDistance, acceptedCount: order.length - unresolved.length,
    unresolvedFaces: Uint32Array.from(unresolved), maximumGraphDistance };
}
