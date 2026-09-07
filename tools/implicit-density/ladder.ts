import { evaluate, frameForBox, gradient, mean, mergeEquivalentFields,
  splitField, volume, type Box, type DensityField, type Vec3 } from "./field";
import { fixtures, type LadderFixture } from "./ladder-fixtures";

export const LADDER_BUDGETS = Object.freeze({
  density: 1e-10, integratedAmount: 1e-10, gradient: 1e-9,
  surfaceResidual: 1e-10, surfaceDisplacement: 1e-9, surfaceNormal: 1e-9, cycles: 100,
});
interface Leaf { readonly box: Box; readonly field: DensityField }
function partition(field: DensityField, box: Box, variant: number): Leaf[] {
  const leaves: Leaf[] = [];
  const centre = frameForBox(box).origin;
  const axis = variant % 3;
  const visit = (f: DensityField, b: Box, depth: number) => {
    // Half-domain finer region, with its axis and side cycling. This exercises
    // query repartitioning, not the production 2:1 topology planner.
    const middle = frameForBox(b).origin;
    const inRegion = variant % 2 ? middle[axis] >= centre[axis] : middle[axis] <= centre[axis];
    if (depth >= (inRegion ? 3 : 1)) { leaves.push({ field: f, box: b }); return; }
    for (const child of splitField(f, b)) visit(child.field, child.box, depth + 1);
  };
  visit(field, box, 0);
  return leaves;
}
function sampleLeaves(leaves: readonly Leaf[], point: Vec3): number {
  const leaf = leaves.find(l => point.every((x, a) => x >= l.box.lower[a] - 1e-14
    && x <= l.box.upper[a] + 1e-14));
  if (!leaf) throw new Error("probe point outside partition");
  return evaluate(leaf.field, point);
}
function fieldGradient(field: DensityField, point: Vec3): Vec3 | null {
  if (field.kind === "polynomial") return gradient(field, point);
  const a = evaluate(field.branches[0], point), b = evaluate(field.branches[1], point);
  if (Math.abs(a - b) < 1e-12) return null;
  const index = field.kind === "minimum" ? +(a > b) : +(a < b);
  return gradient(field.branches[index], point);
}
export function runFixture(fixture: LadderFixture, includeSection = false) {
  let field: DensityField = fixture.field;
  const initialMass = fixture.exactMean(fixture.box) * volume(fixture.box);
  let maximumDensityError = 0, maximumGradientError = 0, maximumSurfaceResidual = 0;
  let maximumSurfaceDisplacement = 0, maximumSurfaceNormalError = 0;
  let maximumMassError = 0, maximumLocalMeanError = 0, maximumSeamJump = 0;
  let maximumLeaves = 0;
  const checkpoints: { cycle: number; leaves: number; massError: number }[] = [];
  for (let cycle = 1; cycle <= LADDER_BUDGETS.cycles; cycle++) {
    const leaves = partition(field, fixture.box, cycle);
    maximumLeaves = Math.max(maximumLeaves, leaves.length);
    const mass = leaves.reduce((sum, l) => sum + mean(l.field, l.box) * volume(l.box), 0);
    maximumMassError = Math.max(maximumMassError, Math.abs(mass - initialMass));
    for (const leaf of leaves) {
      maximumLocalMeanError = Math.max(maximumLocalMeanError,
        Math.abs(mean(leaf.field, leaf.box) - fixture.exactMean(leaf.box)));
      // Query each leaf's faces against the common retained field. Opposing
      // representations must agree even at different local coordinate scales.
      const c = frameForBox(leaf.box).origin;
      for (let axis = 0; axis < 3; axis++) for (const side of [0, 1]) {
        const p = [...c] as [number, number, number];
        p[axis] = side ? leaf.box.upper[axis] : leaf.box.lower[axis];
        maximumSeamJump = Math.max(maximumSeamJump, Math.abs(evaluate(leaf.field, p) - evaluate(field, p)));
      }
    }
    for (const point of fixture.points) maximumDensityError = Math.max(maximumDensityError,
      Math.abs(sampleLeaves(leaves, point) - fixture.exactDensity(point)));
    for (const point of fixture.surfacePoints) maximumSurfaceResidual = Math.max(maximumSurfaceResidual,
      Math.abs(sampleLeaves(leaves, point) - 0.5));
    const merged = mergeEquivalentFields(leaves.map(l => l.field), frameForBox(fixture.box));
    if (!merged) throw new Error(`${fixture.id}: exact represented field refused merge`);
    field = merged;
    if ([1, 10, 100].includes(cycle)) checkpoints.push({ cycle, leaves: leaves.length,
      massError: Math.abs(mass - initialMass) });
  }
  for (const point of fixture.points) {
    const expected = fixture.exactGradient(point), actual = fieldGradient(field, point);
    if (expected && actual) maximumGradientError = Math.max(maximumGradientError,
      ...actual.map((v, a) => Math.abs(v - expected[a])));
    else if (!!expected !== !!actual) throw new Error(`${fixture.id}: lost or invented sharp branch tie`);
  }
  // Intersect the represented field along independently specified analytic
  // normals. A density residual alone has no geometric units and can conceal
  // a displaced surface when the field's gradient is small. At a crease the
  // analytic normal is intentionally undefined, so retain the exact level-set
  // residual check instead of inventing a smoothed normal there.
  let normalProbeCount = 0, creaseProbeCount = 0;
  for (const point of fixture.surfacePoints) {
    const expected = fixture.exactGradient(point);
    if (!expected) { creaseProbeCount++; continue; }
    const length = Math.hypot(...expected);
    const normal = expected.map(v => v / length) as unknown as Vec3;
    const along = (s: number) => point.map((v, a) => v + s * normal[a]) as unknown as Vec3;
    let lo = -.01, hi = .01;
    if (!(evaluate(field, along(lo)) < .5 && evaluate(field, along(hi)) > .5))
      throw new Error(`${fixture.id}: represented surface lost its analytic crossing`);
    for (let iteration = 0; iteration < 60; iteration++) {
      const mid = (lo + hi) / 2;
      if (evaluate(field, along(mid)) < .5) lo = mid; else hi = mid;
    }
    const displacement = (lo + hi) / 2;
    maximumSurfaceDisplacement = Math.max(maximumSurfaceDisplacement, Math.abs(displacement));
    const actual = fieldGradient(field, along(displacement));
    if (!actual) throw new Error(`${fixture.id}: smooth surface acquired a branch tie`);
    const actualLength = Math.hypot(...actual);
    maximumSurfaceNormalError = Math.max(maximumSurfaceNormalError,
      Math.hypot(...actual.map((v, a) => v / actualLength - normal[a])));
    normalProbeCount++;
  }
  const b = LADDER_BUDGETS;
  const passed = maximumDensityError <= b.density && maximumLocalMeanError <= b.density
    && maximumMassError <= b.integratedAmount && maximumGradientError <= b.gradient
    && maximumSurfaceResidual <= b.surfaceResidual && maximumSeamJump <= b.density
    && maximumSurfaceDisplacement <= b.surfaceDisplacement
    && maximumSurfaceNormalError <= b.surfaceNormal;
  const section = includeSection ? (() => {
    const axes = fixture.id.startsWith("edge-")
      ? fixture.id.endsWith("-z") ? [2, 0] : [0, 2] : [0, 1];
    const n = 65, fixed = frameForBox(fixture.box).origin;
    const xs = Array.from({ length: n }, (_, i) => fixture.box.lower[axes[0]] + i / (n - 1) * (fixture.box.upper[axes[0]] - fixture.box.lower[axes[0]]));
    const ys = Array.from({ length: n }, (_, i) => fixture.box.lower[axes[1]] + i / (n - 1) * (fixture.box.upper[axes[1]] - fixture.box.lower[axes[1]]));
    const point = (x: number, y: number): Vec3 => {
      const p = [...fixed] as [number, number, number]; p[axes[0]] = x; p[axes[1]] = y; return p;
    };
    return { x: xs, y: ys, axes, fixed, exact: ys.map(y => xs.map(x => fixture.exactDensity(point(x, y)))),
      represented: ys.map(y => xs.map(x => evaluate(field, point(x, y)))) };
  })() : undefined;
  return { id: fixture.id, description: fixture.description, passed, maximumLeaves,
    maximumDensityError, maximumLocalMeanError, maximumMassError, maximumGradientError,
    maximumSurfaceResidual, maximumSurfaceDisplacement, maximumSurfaceNormalError,
    normalProbeCount, creaseProbeCount, maximumSeamJump, checkpoints, ...(section ? { section } : {}) };
}
export function runLadder(includeSections = false) {
  return fixtures.map(f => runFixture(f, includeSections));
}

/** Deliberately introduce new information in one child: coarsening must fail. */
export function verifyUnrepresentableMerge() {
  const source = fixtures.find(f => f.field.kind === "polynomial")!;
  const children = splitField(source.field, source.box);
  const modified = children.map(c => c.field);
  const first = modified[0];
  if (first.kind !== "polynomial") throw new Error("expected polynomial fixture");
  modified[0] = { ...first, coefficients: first.coefficients.map((x, k) => k ? x : x + 0.01) as unknown as typeof first.coefficients };
  return mergeEquivalentFields(modified, frameForBox(source.box)) === null;
}
