import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  createSparseCM12CurrentMapMeasureWGSL,
  SPARSE_CM12_CURRENT_MAP_GAUSS_4 as gauss,
} from "../lib/methods/adaptive-mass/sparse-cm12-current-map-measure.wgsl";
import { sphereRampMass } from "./helpers/quadratic-pullback-oracle";
import { currentMapMeasureFixtureSampler, type CurrentMapMeasureFixture } from "./helpers/current-map-measure-fixture";

type Point = readonly [number, number, number];
type Value = readonly [number, number, number?];
const clamp = (v: number) => Math.max(0, Math.min(1, v));

/** CPU evaluation of the numerical rule, tested against closed physical
 * integrals below. This measures quadrature accuracy, not GPU wiring. */
function fineMeasure(sample: (p: Point) => Value, lower: Point,
  options: { readonly forcedAxis?: number; readonly fallbackAxes?: boolean } = {}) {
  let evaluations = 0;
  const center = lower.map(v => v + .5);
  let integrationAxis = 1, greatestVariation = -1;
  for (let axis = 0; axis < 3; ++axis) {
    const lo = [...center], hi = [...center];lo[axis] -= .5;hi[axis] += .5;
    const a = sample(lo as unknown as Point), b = sample(hi as unknown as Point);
    const variation = Math.abs((b[2] ?? b[0]) - (a[2] ?? a[0]));
    if (variation > greatestVariation) { greatestVariation = variation;integrationAxis = axis; }
  }
  if (options.forcedAxis!==undefined) integrationAxis=options.forcedAxis;
  const outerA = (integrationAxis + 1) % 3, outerB = (integrationAxis + 2) % 3;
  const rule = (lo: Point, span: number): Value => {
    const result = [0, 0];
    for (let z = 0; z < 4; ++z) for (let x = 0; x < 4; ++x) {
      const at = (t: number): Point => {
        const p = [...lo];p[integrationAxis] += span*t;
        p[outerA] += span*gauss.nodes[x]!;p[outerB] += span*gauss.nodes[z]!;
        return p as unknown as Point;
      };
      const raw = (t: number) => { const v = sample(at(t)); return v[2] ?? v[0]; };
      const splits = [0, 1];
      for (let interval = 0; interval < 8; ++interval) for (const threshold of [0, 1]) {
        let left = interval / 8, right = (interval + 1) / 8;
        const low = raw(left), high = raw(right);
        if ((low < threshold) === (high < threshold)) continue;
        for (let iteration = 0; iteration < 12; ++iteration) {
          const mid = (left + right) / 2;
          if ((raw(mid) < threshold) === (low < threshold)) left = mid; else right = mid;
        }
        splits.push((left + right) / 2);
      }
      splits.sort((a, b) => a - b);
      for (let segment = 0; segment + 1 < splits.length; ++segment) for (let y = 0; y < 4; ++y) {
        const length = splits[segment + 1]! - splits[segment]!;
        const value = sample(at(splits[segment]! + length * gauss.nodes[y]!));
        const weight = length * span ** 3 * gauss.weights[x]! * gauss.weights[y]! * gauss.weights[z]!;
        result[0] += weight * value[0]; result[1] += weight * value[1]; ++evaluations;
      }
    }
    return result as unknown as Value;
  };
  const stack = [{ lower, span: 1, depth: 0, coarse: rule(lower, 1) }];
  const value = [0, 0], error = [0, 0];
  while (stack.length) {
    const box = stack.pop()!, span = box.span / 2;
    const children = Array.from({ length: 8 }, (_, i) => {
      const lower: Point = [box.lower[0] + span * (i & 1), box.lower[1] + span * ((i >> 1) & 1),
        box.lower[2] + span * (i >> 2)];
      return { lower, span, depth: box.depth + 1, coarse: rule(lower, span) };
    });
    const high = [0, 1].map(axis => children.reduce((sum, child) => sum + child.coarse[axis]!, 0));
    const difference = high.map((v, axis) => Math.abs(v - box.coarse[axis]!));
    if (box.depth < 3 && difference.some((e, axis) => e > 1e-4 * box.span ** 3 + 1e-4 * Math.abs(high[axis]!))) {
      stack.push(...children);
    } else {
      for (let axis = 0; axis < 2; ++axis) { value[axis] += high[axis]!; error[axis] += difference[axis]!; }
    }
  }
  const resolved = error.every((e, axis) => e <= 1e-4 + 1e-4 * Math.abs(value[axis]!));
  if (!resolved&&options.fallbackAxes!==false&&options.forcedAxis===undefined) {
    for (let offset = 1; offset < 3; ++offset) {
      try { return fineMeasure(sample,lower,{ forcedAxis:(integrationAxis+offset)%3,fallbackAxes:false }); }
      catch (error) { if (!(error instanceof assert.AssertionError)) throw error; }
    }
  }
  assert.ok(resolved,
    `Unresolved estimate ${error} at ${lower}`);
  return { value, error, evaluations };
}

test("four-point current-map measure rule integrates all degree-seven monomials", () => {
  for (let degree = 0; degree <= 7; ++degree) {
    const actual = gauss.nodes.reduce((sum, x, i) => sum + gauss.weights[i]! * x ** degree, 0);
    assert.ok(Math.abs(actual - 1 / (degree + 1)) < 2e-16, `${degree}: ${actual}`);
  }
});

test("clamped plane integral converges without fitting a native mean", () => {
  for (const intercept of [.137, .331, .793]) {
    const { value } = fineMeasure(([,x]) => [clamp(intercept - x), 2 * clamp(intercept - x), intercept-x], [0, 0, 0]);
    assert.ok(Math.abs(value[0]! - intercept ** 2 / 2) < 3e-6, `${intercept}: ${value[0]! - intercept ** 2 / 2}`);
    assert.ok(Math.abs(value[1]! - intercept ** 2) < 6e-6);
  }
});

test("translated and sheared curved measures conserve analytic mass and momentum", (t) => {
  const radius = 1.4, width = 1, center: Point = [2.37, 2.19, 2.63];
  const expectedMass = sphereRampMass(radius, width);
  // The shear is a non-rigid volume-preserving spatial map, and the momentum
  // varies with y. Constant velocity or a translated display-only sphere
  // cannot satisfy this test's density-plus-momentum quadrature contract.
  for (const shear of [0, .17]) {
    let mass = 0, momentum = 0, evaluations = 0;
    for (let z = 0; z < 6; ++z) for (let y = 0; y < 6; ++y) for (let x = 0; x < 6; ++x) {
      const result = fineMeasure(([px, py, pz]) => {
        const dx = px - shear * py - center[0], dy = py - center[1], dz = pz - center[2];
        const raw = .5 - (dx * dx + dy * dy + dz * dz - radius ** 2) / (2 * radius * width);
        const density = clamp(raw);
        return [density, density * .7 * py, raw];
      }, [x, y, z]);
      mass += result.value[0]!; momentum += result.value[1]!; evaluations += result.evaluations;
    }
    assert.ok(Math.abs(mass / expectedMass - 1) < 1e-5, `shear ${shear}: mass ${mass}, oracle ${expectedMass}`);
    assert.ok(Math.abs(momentum / (.7 * center[1] * expectedMass) - 1) < 1e-5,
      `shear ${shear}: momentum ${momentum}`);
    assert.ok(evaluations < 5_000_000, `bounded small-domain quadrature work: ${evaluations}`);
    t.diagnostic(`shear=${shear}, relative mass error=${mass / expectedMass - 1}, relative momentum error=${momentum / (.7 * center[1] * expectedMass) - 1}, integrand samples=${evaluations}`);
  }
});

test("current-map measure factory rejects invalid storage and unbounded quadrature", () => {
  assert.throws(() => createSparseCM12CurrentMapMeasureWGSL({ baseWords: -1, dimensions: [1, 1, 1] }));
  assert.throws(() => createSparseCM12CurrentMapMeasureWGSL({ baseWords: 0, dimensions: [0, 1, 1] }));
  assert.throws(() => createSparseCM12CurrentMapMeasureWGSL({ baseWords: 0, dimensions: [1, 1, 1], maximumRefinementDepth: 6 }));
  assert.throws(() => createSparseCM12CurrentMapMeasureWGSL({ baseWords: 0, dimensions: [1, 1, 1], absoluteTolerance: NaN }));
  assert.throws(() => createSparseCM12CurrentMapMeasureWGSL({ baseWords: 0, dimensions: [1, 1, 1], unchangedSupportHook: "bad();" }));
});

test("production-scale sphere cap integrates falling momentum at unchanged tolerance", (t) => {
  const center: Point = [16, 18.25 - .3268883466720581 / .05 / 30, 16];
  const velocity = -.3268883466720581 / .05;
  for (const lower of [[10, 18, 16], [21, 18, 16], [16, 18, 10]] as const) {
    const result = fineMeasure(([x, y, z]) => {
      const raw = .5 - ((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 - 25) / 10;
      const density = clamp(raw);return [density, density * velocity, raw];
    }, lower);
    t.diagnostic(`cap ${lower}, value=${result.value}, estimate=${result.error}, samples=${result.evaluations}`);
  }
});

test("all production-scale falling-sphere supports meet the original integration tolerance", (t) => {
  const radius = 5, center: Point = [16, 18.03207443555196, 16];
  const velocity = -.3268883466720581 / .05;
  let mass = 0, momentum = 0, maximumDensityError = 0, maximumMomentumError = 0;
  for (let z = 10; z < 22; ++z) for (let y = 12; y < 24; ++y) for (let x = 10; x < 22; ++x) {
    const result = fineMeasure(([px, py, pz]) => {
      const raw = .5 - ((px-center[0]) ** 2+(py-center[1]) ** 2+(pz-center[2]) ** 2-radius ** 2)/(2*radius);
      const density = clamp(raw);return [density, density*velocity, raw];
    }, [x, y, z]);
    mass += result.value[0]!;momentum += result.value[1]!;
    maximumDensityError = Math.max(maximumDensityError, result.error[0]!);
    maximumMomentumError = Math.max(maximumMomentumError, result.error[1]!);
  }
  const expected = sphereRampMass(radius, 1);
  assert.ok(Math.abs(mass/expected-1) < 1e-5, `mass ${mass} vs ${expected}`);
  assert.ok(Math.abs(momentum/(velocity*expected)-1) < 1e-5);
  t.diagnostic(`production sphere relative mass error=${mass/expected-1}, maximum local estimates=${maximumDensityError},${maximumMomentumError}`);
});

test("archived near-impact density retries a better slicing axis independently of old trajectory numerics", (t) => {
  const fixture = JSON.parse(readFileSync(new URL("./fixtures/current-map-measure-step7.json",import.meta.url),"utf8")) as CurrentMapMeasureFixture;
  const sample = currentMapMeasureFixtureSampler(fixture),lower = fixture.cellLowerFine as unknown as Point;
  assert.throws(() => fineMeasure(sample,lower,{ fallbackAxes:false }),/Unresolved estimate/);
  const result = fineMeasure(sample,lower);
  assert.ok(Math.abs(result.value[0]!-fixture.referenceMean[0]!)<1e-6);
  assert.ok(Math.abs(result.value[1]!-fixture.referenceMean[1]!)<1e-5);
  t.diagnostic(`actual candidate mean=${result.value}, error estimate=${result.error}`);
});

test("momentum consumes the same certified increment as the candidate density", () => {
  const shader = createSparseCM12CurrentMapMeasureWGSL({ baseWords:0,dimensions:[1,1,1] });
  const sample = shader.slice(shader.indexOf("fn cm12CurrentMapMeasureSample("),
    shader.indexOf("fn cm12CurrentMapMeasureOrder("));
  assert.match(sample,/cm12CurrentMapDensityAtFine\(point,bank\)/);
  assert.match(sample,/cm12CurrentMapEvaluateIncrement\(point,bank\)\.point/);
  assert.match(sample,/cm12CurrentMapVelocity\(departure\)/);
  assert.doesNotMatch(sample,/cm12CurrentMapDeparture\(/,
    "retracing raw RK2 would make momentum use a different pullback than density");
});
