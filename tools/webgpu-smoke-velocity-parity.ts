export interface VelocityParityMetrics {
  readonly comparedCells: number;
  readonly weightedRelativeL2: number;
  readonly cosineSimilarity: number;
  readonly candidateToReferenceEnergyRatio: number;
  readonly candidateToReferencePeakRatio: number;
}

export interface VelocityParityLimits {
  readonly maximumWeightedRelativeL2: number;
  readonly minimumCosineSimilarity: number;
  readonly minimumEnergyRatio: number;
  readonly maximumEnergyRatio: number;
  readonly minimumPeakRatio: number;
  readonly maximumPeakRatio: number;
}

export const DAM_BREAK_VELOCITY_PARITY_LIMITS: VelocityParityLimits = Object.freeze({
  maximumWeightedRelativeL2: 1,
  minimumCosineSimilarity: 0.5,
  minimumEnergyRatio: 0.25,
  maximumEnergyRatio: 4,
  minimumPeakRatio: 0.5,
  maximumPeakRatio: 2,
});

export interface CompactVelocityRaster {
  readonly field: Float32Array;
  readonly coveredCells: number;
  readonly overlapCells: number;
  readonly invalidRows: number;
}

/**
 * Expand one xyz velocity per adaptive power leaf onto the finest cubic QA
 * lattice. `headers` uses the live 48-byte power-leaf ABI: word 0 is the
 * linear finest-cell origin and word 3 is the cubic leaf size. `velocities`
 * uses the live vec4 ABI whose w lane is one only for a solved reconstruction.
 *
 * This is deliberately a CPU readback adapter, not a dense simulation
 * publication. Missing, overlapping, out-of-bounds, or unsolved leaves write
 * NaNs so the parity gate fails closed instead of silently scoring zeros.
 */
export function rasterizeCompactPowerCellVelocities(
  headers: ArrayLike<number>,
  velocities: ArrayLike<number>,
  rowCount: number,
  dimensions: readonly [number, number, number],
): CompactVelocityRaster {
  const [nx, ny, nz] = dimensions;
  if (!Number.isSafeInteger(rowCount) || rowCount < 0) throw new RangeError("Power velocity row count must be non-negative");
  if (![nx, ny, nz].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("Power velocity dimensions must be positive integers");
  }
  if (headers.length < rowCount * 12 || velocities.length < rowCount * 4) {
    throw new RangeError("Power velocity buffers do not contain every declared row");
  }
  const cellCount = nx * ny * nz;
  const field = new Float32Array(cellCount * 3);
  field.fill(Number.NaN);
  const owners = new Int32Array(cellCount);
  owners.fill(-1);
  let coveredCells = 0, overlapCells = 0, invalidRows = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const cell = Number(headers[row * 12]) >>> 0;
    const size = Number(headers[row * 12 + 3]) >>> 0;
    const origin: [number, number, number] = [cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny))];
    const solved = Number(velocities[row * 4 + 3]) === 1;
    const vector = [Number(velocities[row * 4]), Number(velocities[row * 4 + 1]), Number(velocities[row * 4 + 2])];
    const valid = size > 0 && cell < cellCount
      && origin[0] + size <= nx && origin[1] + size <= ny && origin[2] + size <= nz
      && solved && vector.every(Number.isFinite);
    if (!valid) invalidRows += 1;
    if (size === 0 || cell >= cellCount || origin[0] + size > nx || origin[1] + size > ny || origin[2] + size > nz) continue;
    for (let z = origin[2]; z < origin[2] + size; z += 1) {
      for (let y = origin[1]; y < origin[1] + size; y += 1) {
        for (let x = origin[0]; x < origin[0] + size; x += 1) {
          const index = x + nx * (y + ny * z);
          if (owners[index] >= 0) {
            overlapCells += 1;
            field[3 * index] = Number.NaN;
            field[3 * index + 1] = Number.NaN;
            field[3 * index + 2] = Number.NaN;
            continue;
          }
          owners[index] = row;
          coveredCells += 1;
          if (!valid) continue;
          field[3 * index] = vector[0];
          field[3 * index + 1] = vector[1];
          field[3 * index + 2] = vector[2];
        }
      }
    }
  }
  return { field, coveredCells, overlapCells, invalidRows };
}

/**
 * Compare two collocated xyz velocity fields over their shared liquid support.
 * Occupancy weights are the minimum of the two represented liquid fractions,
 * so a plausible velocity in unrelated air cells cannot improve the score.
 */
export function compareVelocityFields(
  candidate: ArrayLike<number>,
  reference: ArrayLike<number>,
  candidateVolume: ArrayLike<number>,
  referenceVolume: ArrayLike<number>,
): VelocityParityMetrics {
  if (candidate.length !== reference.length || candidate.length % 3 !== 0) {
    throw new RangeError("Velocity parity fields must have matching xyz components");
  }
  const cells = candidate.length / 3;
  if (candidateVolume.length !== cells || referenceVolume.length !== cells) {
    throw new RangeError("Velocity parity occupancy must match the field dimensions");
  }
  let error2 = 0, candidate2 = 0, reference2 = 0, dot = 0;
  let candidatePeak = 0, referencePeak = 0, comparedCells = 0;
  for (let cell = 0; cell < cells; cell += 1) {
    const weight = Math.max(0, Math.min(1, Math.min(candidateVolume[cell], referenceVolume[cell])));
    if (!(weight > 1e-4)) continue;
    comparedCells += 1;
    let localCandidate2 = 0, localReference2 = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      const c = candidate[3 * cell + axis], r = reference[3 * cell + axis];
      if (!Number.isFinite(c) || !Number.isFinite(r)) {
        return { comparedCells, weightedRelativeL2: Infinity, cosineSimilarity: -1,
          candidateToReferenceEnergyRatio: Infinity, candidateToReferencePeakRatio: Infinity };
      }
      error2 += weight * (c - r) ** 2;
      candidate2 += weight * c * c; reference2 += weight * r * r; dot += weight * c * r;
      localCandidate2 += c * c; localReference2 += r * r;
    }
    candidatePeak = Math.max(candidatePeak, Math.sqrt(localCandidate2));
    referencePeak = Math.max(referencePeak, Math.sqrt(localReference2));
  }
  return {
    comparedCells,
    weightedRelativeL2: Math.sqrt(error2 / Math.max(reference2, 1e-30)),
    cosineSimilarity: dot / Math.sqrt(Math.max(candidate2 * reference2, 1e-30)),
    candidateToReferenceEnergyRatio: candidate2 / Math.max(reference2, 1e-30),
    candidateToReferencePeakRatio: candidatePeak / Math.max(referencePeak, 1e-30),
  };
}

export function velocityParityFailures(
  metrics: VelocityParityMetrics,
  limits: VelocityParityLimits = DAM_BREAK_VELOCITY_PARITY_LIMITS,
): readonly string[] {
  const failures: string[] = [];
  if (metrics.comparedCells === 0) failures.push("no shared liquid velocity cells");
  if (!(metrics.weightedRelativeL2 <= limits.maximumWeightedRelativeL2)) failures.push("velocity relative L2");
  if (!(metrics.cosineSimilarity >= limits.minimumCosineSimilarity)) failures.push("velocity direction cosine");
  if (!(metrics.candidateToReferenceEnergyRatio >= limits.minimumEnergyRatio
    && metrics.candidateToReferenceEnergyRatio <= limits.maximumEnergyRatio)) failures.push("velocity energy ratio");
  if (!(metrics.candidateToReferencePeakRatio >= limits.minimumPeakRatio
    && metrics.candidateToReferencePeakRatio <= limits.maximumPeakRatio)) failures.push("velocity peak ratio");
  return failures;
}
