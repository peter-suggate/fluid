/** Geometry acceptance probe, deliberately independent of mass/positivity checks.
 * There is no production analytic initializer for this experimental basis.
 * Audit explicit sampled and exact-isosurface initializers, then the proposed mean correction.
 * Exit nonzero only with --assert-geometry: default emits the failure evidence.
 */
import { bernsteinCellMeans, compileBernsteinSupport, evaluateBernsteinCell,
  fitPositiveBernsteinMeans, positiveBernsteinField } from "./implicit-density/positive-bernstein";

const fixtures = [
  ...[.25, .3, .75].map(fill => ({ id: `axis-${fill}`, mean: fill, height: () => fill })),
  { id: "oblique", mean: .5, height: (x: number, z: number) => .3 + .3 * x + .1 * z },
  { id: "shallow-curve", mean: .3 + .2 / 6, height: (x: number, z: number) => .3 + .2 * ((x - .5) ** 2 + (z - .5) ** 2) },
  { id: "crease", mean: .3 + .2 / 4, height: (x: number) => .3 + .2 * Math.abs(x - .5) },
];
const support = compileBernsteinSupport([{ lower: [0, 0, 0], width: 1 }]);
const tolerance = .01; // One percent of support width, not a mass tolerance.
export const receipts = fixtures.flatMap(fixture => ["indicator", "unit-ramp", ...(fixture.id === "crease" ? [] : ["exact-iso-polynomial"])].map(initializer => {
  const seed = positiveBernsteinField(support, support.positions.map(([x, y, z]) => {
    const signed = fixture.height(x, z) - y;
    if (initializer === "exact-iso-polynomial") {
      // A nonnegative quadratic with EXACT source isosurface before fitting.
      // Convert sampled x²/z² values to Bernstein controls at midpoints.
      const correction = fixture.id === "shallow-curve"
        ? .6 * .2 / 4 * (Number(x === .5) + Number(z === .5)) : 0;
      return .5 + .6 * signed - correction;
    }
    return initializer === "indicator" ? Number(signed >= 0) : Math.max(0, Math.min(1, .5 + signed));
  }));
  let seedSourceIsoResidual = 0;
  for (let xi = 0; xi <= 10; xi++) for (let zi = 0; zi <= 10; zi++) {
    const x = xi / 10, z = zi / 10;
    seedSourceIsoResidual = Math.max(seedSourceIsoResidual,
      Math.abs(evaluateBernsteinCell(seed, 0, [x, fixture.height(x, z), z]) - .5));
  }
  if (initializer === "exact-iso-polynomial" && seedSourceIsoResidual > 1e-13) throw new Error("Invalid exact predictor");
  const fit = fitPositiveBernsteinMeans(seed, [fixture.mean]);
  let maximumHeightError = 0, maximumSourceIsoResidual = 0, wrongCrossingCount = 0, missingColumns = 0;
  let worst: unknown;
  let minimumDensity = Infinity, maximumDensity = -Infinity;
  // Include faces and corners: the private center correction vanishes there.
  for (let xi = 0; xi <= 10; xi++) for (let zi = 0; zi <= 10; zi++) {
    const x = xi / 10, z = zi / 10, expected = fixture.height(x, z);
    maximumSourceIsoResidual = Math.max(maximumSourceIsoResidual,
      Math.abs(evaluateBernsteinCell(fit.field, 0, [x, expected, z]) - .5));
    const roots: number[] = [];
    let low = evaluateBernsteinCell(fit.field, 0, [x, 0, z]) - .5;
    for (let yi = 1; yi <= 512; yi++) {
      const y = yi / 512, high = evaluateBernsteinCell(fit.field, 0, [x, y, z]) - .5;
      minimumDensity = Math.min(minimumDensity, high + .5); maximumDensity = Math.max(maximumDensity, high + .5);
      if (low === 0 || high * low < 0) {
        let a = (yi - 1) / 512, b = y;
        for (let iteration = 0; iteration < 40; iteration++) {
          const middle = (a + b) / 2;
          if ((evaluateBernsteinCell(fit.field, 0, [x, middle, z]) - .5) * low > 0) a = middle;
          else b = middle;
        }
        roots.push((a + b) / 2);
      }
      low = high;
    }
    if (roots.length !== 1) wrongCrossingCount++;
    if (!roots.length) missingColumns++;
    for (const root of roots) {
      const error = Math.abs(root - expected);
      if (error > maximumHeightError) { maximumHeightError = error; worst = { x, z, expected, roots }; }
    }
  }
  const massError = Math.abs(bernsteinCellMeans(fit.field)[0] - fixture.mean);
  return { fixture: fixture.id, initializer, seedSourceIsoResidual, massError, minimumDensity, maximumDensity,
    maximumHeightError, maximumSourceIsoResidual, wrongCrossingCount, missingColumns,
    maximumInteriorChange: fit.maximumInteriorChange, maximumTraceChange: fit.maximumTraceChange,
    acceptable: maximumHeightError < tolerance && wrongCrossingCount === 0, worst };
}));
console.log(JSON.stringify({ supportWidth: 1, isovalue: .5, heightTolerance: tolerance,
  claim: "Experimental mean correction; these seed conventions are explicit hypotheses, not production initializers", receipts }, null, 2));
if (process.argv.includes("--assert-geometry") && receipts.some(receipt => !receipt.acceptable)) process.exitCode = 1;
