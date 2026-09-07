/** Independent authored equations for the four production partition fixtures.
 * Deliberately imports neither the scene compiler nor a production evaluator.
 */
export type PartitionFixture = "flat" | "quadratic" | "sphere-pool" | "sharp-box";
type Point = readonly [number, number, number];
export function partitionAnalyticPhi(fixture: PartitionFixture, [x, y, z]: Point): number {
  if (fixture === "flat") return y - .8 * .43;
  if (fixture === "quadratic") return y - (.32 + .6 * x * x + .35 * z * z);
  if (fixture === "sphere-pool") return Math.min(y - .2,
    ((x - .03) ** 2 + (y - .51) ** 2 + (z + .02) ** 2 - .14 ** 2) / (2 * .14));
  return Math.min(y - .2, Math.max(-.17 - x, x - .13, .16 - y, y - .57, -.13 - z, z - .18));
}

function exactRoots(fixture: PartitionFixture, x: number, z: number): number[] {
  if (fixture === "flat") return [.8 * .43];
  if (fixture === "quadratic") return [.32 + .6 * x * x + .35 * z * z];
  if (fixture === "sharp-box") return [x > -.17 && x < .13 && z > -.13 && z < .18 ? .57 : .2];
  const squaredHeight = .14 ** 2 - (x - .03) ** 2 - (z + .02) ** 2;
  return squaredHeight > 0 ? [.2, .51 - Math.sqrt(squaredHeight), .51 + Math.sqrt(squaredHeight)] : [.2];
}

// Physical distance for the sphere; vertical residual is a conservative
// distance bound for a height graph. Box crossings lie on vertical rays with
// fixed x/z, so its L-infinity branch supplies the relevant closest-face error.
function zeroDistance(fixture: PartitionFixture, p: Point): number {
  if (fixture === "sphere-pool") return Math.abs(Math.min(p[1] - .2,
    Math.hypot(p[0] - .03, p[1] - .51, p[2] + .02) - .14));
  return Math.abs(partitionAnalyticPhi(fixture, p));
}

function roots(values: number[], h: number) {
  const result: { height: number; lower: number }[] = [];
  for (let y = 0; y + 1 < values.length; y++) {
    const a = values[y]!, b = values[y + 1]!;
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === b || a * b > 0) continue;
    const height = (y + .5 - a / (b - a)) * h;
    if (!result.length || Math.abs(height - result[result.length - 1]!.height) > 1e-12)
      result.push({ height, lower: y });
  }
  return result;
}

export function partitionAnalyticBudgets(fixture: PartitionFixture, h: number) {
  return {
    // Nearest binary16 rounding plus float32 coefficient/arithmetic allowance.
    sample: (phi: number) => Math.abs(phi) / 2048 + 1e-6,
    // A L-infinity box ray can switch active face between adjacent samples.
    // Plane/height roots are affine in y; the sphere incurs quadratic sampling.
    surface_m: fixture === "sharp-box" ? h / 2 + 5e-5
      : fixture === "sphere-pool" ? h * h / (4 * .14) + 5e-5 : 5e-5,
  };
}

export function measurePartitionAnalyticField(fixture: PartitionFixture, phi: Float32Array,
  dimensions: readonly [number, number, number], h: number) {
  const [nx, ny, nz] = dimensions;
  if (phi.length !== nx * ny * nz) throw new Error("partition oracle requires the complete fine lattice");
  const budget = partitionAnalyticBudgets(fixture, h);
  let sampleCount = 0, missingSamples = 0, maximumSamplePrecisionRatio = 0;
  let analyticCrossings = 0, sampledReferenceCrossings = 0, observedCrossings = 0;
  let unresolvedAnalyticColumns = 0, changedCrossingColumns = 0;
  let maximumRootPrecisionRatio = 0, maximumZeroDistance_m = 0;
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    const wx = -.4 + (x + .5) * h, wz = -.4 + (z + .5) * h;
    const expected: number[] = [], actual: number[] = [];
    for (let y = 0; y < ny; y++) {
      const value = phi[x + nx * (y + ny * z)]!;
      const reference = partitionAnalyticPhi(fixture, [wx, (y + .5) * h, wz]);
      expected.push(reference); actual.push(value);
      // Inspect the union of both interface bands: missing expected support
      // and a newly invented near-zero interface both fail independently.
      if (Math.abs(reference) <= 1.5 * h || Math.abs(value) <= 1.5 * h) {
        sampleCount++;
        if (!Number.isFinite(value)) missingSamples++;
        else maximumSamplePrecisionRatio = Math.max(maximumSamplePrecisionRatio,
          Math.abs(value - reference) / budget.sample(reference));
      }
    }
    const analytic = exactRoots(fixture, wx, wz), ideal = roots(expected, h), measured = roots(actual, h);
    analyticCrossings += analytic.length;
    sampledReferenceCrossings += ideal.length;
    observedCrossings += measured.length;
    if (ideal.length !== analytic.length) unresolvedAnalyticColumns++;
    if (measured.length !== ideal.length) changedCrossingColumns++;
    else measured.forEach((root, i) => {
      const reference = ideal[i]!, a = expected[reference.lower]!, b = expected[reference.lower + 1]!;
      const uncertainty = budget.sample(a) + budget.sample(b);
      // Root sensitivity is derived from that ray's slope, rather than a
      // fixed vertical tolerance that would reject near-tangent sphere rays.
      const allowance = h * uncertainty / Math.max(Math.abs(b - a) - uncertainty, 1e-12);
      maximumRootPrecisionRatio = Math.max(maximumRootPrecisionRatio,
        Math.abs(root.height - reference.height) / allowance);
      maximumZeroDistance_m = Math.max(maximumZeroDistance_m,
        zeroDistance(fixture, [wx, root.height, wz]));
    });
  }
  return { sampleCount, missingSamples, maximumSamplePrecisionRatio, analyticCrossings,
    sampledReferenceCrossings, observedCrossings, unresolvedAnalyticColumns,
    changedCrossingColumns, maximumRootPrecisionRatio, maximumZeroDistance_m,
    surfaceBudget_m: budget.surface_m };
}
