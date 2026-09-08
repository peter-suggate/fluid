/** QA-only native-centre velocity diagnostics. No reconstructed fluid field. */
export type Point3 = readonly [number, number, number];

/** Captures run before frame commit flips the scalar and face parity words.
 * Collocation only republishes wet effective velocities; its dry entries can
 * still contain earlier VEX/gather values and are not projected halo samples. */
export function selectSnapshotVelocity(stage: string, scalarParity: number, faceParity: number) {
  if (![scalarParity, faceParity].every(value => value === 0 || value === 1))
    throw new Error("Invalid frame parity");
  if (stage === "transport-velocity-extension") return {
    densityParity: scalarParity, velocityPlane: "effective", extended: true,
  };
  if (stage === "velocity-projection") return {
    densityParity: scalarParity ^ 1,
    velocityPlane: (faceParity ^ 1) ? "cellVelocityB" : "cellVelocityA", extended: false,
  };
  throw new Error(`Unsupported velocity snapshot stage: ${stage}`);
}
export interface NativeVelocitySample {
  readonly id: number;
  readonly point: Point3;
  readonly velocity: Point3;
  readonly volume: number;
}

function solve(matrix: number[][], rhs: number[]): number[] | undefined {
  const a = matrix.map((row, i) => [...row, rhs[i]!]);
  for (let col = 0; col < rhs.length; col++) {
    let pivot = col;
    for (let row = col + 1; row < rhs.length; row++)
      if (Math.abs(a[row]![col]!) > Math.abs(a[pivot]![col]!)) pivot = row;
    if (Math.abs(a[pivot]![col]!) < 1e-20) return undefined;
    [a[pivot], a[col]] = [a[col]!, a[pivot]!];
    const divisor = a[col]![col]!;
    for (let j = col; j <= rhs.length; j++) a[col]![j]! /= divisor;
    for (let row = 0; row < rhs.length; row++) if (row !== col) {
      const factor = a[row]![col]!;
      for (let j = col; j <= rhs.length; j++) a[row]![j]! -= factor * a[col]![j]!;
    }
  }
  return a.map(row => row[rhs.length]!);
}

export function gradientInvariants(gradient: readonly number[]) {
  const strain = gradient.map((value, i) => .5 * (value + gradient[(i % 3) * 3 + Math.floor(i / 3)]!));
  return { gradient_per_s: [...gradient], strainFrobenius_per_s: Math.hypot(...strain),
    divergence_per_s: gradient[0]! + gradient[4]! + gradient[8]!,
    vorticity_per_s: [gradient[7]! - gradient[5]!, gradient[2]! - gradient[6]!, gradient[3]! - gradient[1]!] };
}

/** Volume-weighted least-squares affine velocity; exact for affine samples. */
export function analyzeNativeVelocity(samples: readonly NativeVelocitySample[], expected: Point3) {
  if (!samples.length) return { count: 0 };
  const volume = samples.reduce((sum, cell) => sum + cell.volume, 0);
  const mean = [0, 0, 0], center = [0, 0, 0];
  for (const cell of samples) for (let axis = 0; axis < 3; axis++) {
    mean[axis]! += cell.volume * cell.velocity[axis]! / volume;
    center[axis]! += cell.volume * cell.point[axis]! / volume;
  }
  const covariance = Array.from({ length: 3 }, () => [0, 0, 0]);
  const cross = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (const cell of samples) {
    const q = cell.point.map((value, axis) => value - center[axis]!);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      covariance[i]![j]! += cell.volume * q[i]! * q[j]!;
      cross[i]![j]! += cell.volume * (cell.velocity[i]! - mean[i]!) * q[j]!;
    }
  }
  const rows = cross.map(rhs => solve(covariance, rhs));
  const gradient = rows.every(Boolean) ? rows.flatMap(row => row!) : undefined;
  let maxSpeed = 0, maxUniformResidual = 0, maxExpectedResidual = 0, maxAffineResidual = 0;
  let uniformSquare = 0, affineSquare = 0;
  let worstUniform: unknown, worstAffine: unknown;
  for (const cell of samples) {
    const constantResidual = Math.hypot(...cell.velocity.map((value, axis) => value - mean[axis]!));
    const expectedResidual = Math.hypot(...cell.velocity.map((value, axis) => value - expected[axis]!));
    const affine = mean.map((value, axis) => value + (gradient ? [0, 1, 2].reduce((sum, j) =>
      sum + gradient[3 * axis + j]! * (cell.point[j]! - center[j]!), 0) : 0));
    const affineResidual = Math.hypot(...cell.velocity.map((value, axis) => value - affine[axis]!));
    maxSpeed = Math.max(maxSpeed, Math.hypot(...cell.velocity));
    maxExpectedResidual = Math.max(maxExpectedResidual, expectedResidual);
    if (constantResidual >= maxUniformResidual) { maxUniformResidual = constantResidual; worstUniform = cell; }
    if (affineResidual >= maxAffineResidual) { maxAffineResidual = affineResidual; worstAffine = cell; }
    uniformSquare += cell.volume * constantResidual ** 2;
    affineSquare += cell.volume * affineResidual ** 2;
  }
  return { count: samples.length, volume_m3: volume, center_m: center, meanVelocity_m_s: mean,
    expectedTranslationVelocity_m_s: expected, maximumSpeed_m_s: maxSpeed,
    maximumExpectedTranslationResidual_m_s: maxExpectedResidual,
    maximumUniformResidual_m_s: maxUniformResidual, uniformRmsResidual_m_s: Math.sqrt(uniformSquare / volume),
    maximumAffineResidual_m_s: gradient ? maxAffineResidual : null,
    affineRmsResidual_m_s: gradient ? Math.sqrt(affineSquare / volume) : null,
    affine: gradient ? gradientInvariants(gradient) : null, worstUniform, worstAffine };
}

/** Face-connected native-centre differences, with physical face area / distance²
 * weights. Full 3D fitting handles tangential offsets at coarse/fine faces;
 * finite differences of an expanded piecewise-constant display are not used. */
export function nativeLocalGradient(cell: NativeVelocitySample,
  neighbors: readonly { sample: NativeVelocitySample; area: number }[]) {
  const matrix = Array.from({ length: 3 }, () => [0, 0, 0]);
  const cross = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (const neighbor of neighbors) {
    const d = neighbor.sample.point.map((value, axis) => value - cell.point[axis]!);
    const weight = neighbor.area / d.reduce((sum, value) => sum + value * value, 0);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      matrix[i]![j]! += weight * d[i]! * d[j]!;
      cross[i]![j]! += weight * (neighbor.sample.velocity[i]! - cell.velocity[i]!) * d[j]!;
    }
  }
  const rows = cross.map(rhs => solve(matrix, rhs));
  return rows.every(Boolean) ? gradientInvariants(rows.flatMap(row => row!)) : undefined;
}
