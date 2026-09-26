import assert from "node:assert/strict";
import test from "node:test";
import { gradedSeamLayout, seamLayout } from "./helpers/uniform-geometric-seam";
import { mixedPressureFixture, faceGradient, geometricDivergence, adjointDivergence, factorPressureCore, deferredPressureStep, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";

const max = (values: Iterable<number>) => Math.max(...Array.from(values, Math.abs));
test("pressure seam reconstruction preserves affine pressure on anisotropic graded faces", () => {
  for (const layout of mixedPressureLayouts()) {
    const fixture = mixedPressureFixture(layout);
    for (const gradient of [[0, 0, 0], [1, 0, 0], [0, -9.81, 0], [0, 0, 1], [2, -3, .25]]) {
      const pressure = fixture.cells.map(c => 17 + c.center.reduce((s, v, a) => s + v * gradient[a]!, 0));
      const actual = faceGradient(fixture, pressure);
      assert.ok(max(actual.map((v, k) => v - gradient[fixture.faces[k]!.axis]!)) < 1e-11);
    }
  }
});

test("both uniform endpoints retain the original two-point gradient and geometric adjoint", () => {
  for (const mode of ["fine", "coarse"] as const) {
    const fixture = mixedPressureFixture(seamLayout(0, mode));
    assert.ok(fixture.gradients.every(row => row.size === 2));
    const velocity = fixture.faces.map((_, k) => Math.sin(k * 13));
    const geometric = geometricDivergence(fixture, velocity), adjoint = adjointDivergence(fixture, velocity);
    assert.ok(max(geometric.map((v, i) => v - adjoint[i]!)) < 1e-12);
  }
});

test("reconstruction changes adjointness: a transpose cannot replace physical divergence", () => {
  const fixture = mixedPressureFixture(gradedSeamLayout(0));
  const velocity = fixture.faces.map((_, k) => Math.sin(k * 13));
  const geometric = geometricDivergence(fixture, velocity), adjoint = adjointDivergence(fixture, velocity);
  // The reconstructed physical operator is nonsymmetric. Defining a transpose
  // would make it symmetric but would change the meaning of physical fluxes.
  assert.ok(max(geometric.map((v, i) => v - adjoint[i]!)) > .01);
  for (const divergence of [geometric, adjoint]) {
    assert.ok(Math.abs(divergence.reduce((s, v, i) => s + v * fixture.cells[i]!.volume, 0)) < 1e-10);
  }
  const p = fixture.cells.map((_, i) => Math.cos(i * 7));
  const gp = faceGradient(fixture, p);
  const facePairing = gp.reduce((s, v, k) => s + v * velocity[k]! * fixture.faces[k]!.area * fixture.faces[k]!.distance, 0);
  const cellPairing = adjoint.reduce((s, v, i) => s + p[i]! * v * fixture.cells[i]!.volume, 0);
  assert.ok(Math.abs(facePairing + cellPairing) < 1e-10);
});

test("deferred seam correction converges on full 3D boxes with physical fluxes and the symmetric pressure core", () => {
  const layouts = mixedPressureLayouts().filter(layout => layout.lattice.dimensions.every(n => n >= 8));
  for (const [index, layout] of layouts.entries()) {
    const fixture = mixedPressureFixture(layout), solve = factorPressureCore(fixture);
    for (const mode of ["hydrostatic", "random"] as const) {
      const target = fixture.cells.map((c, i) => mode === "hydrostatic" ? -9.81 * c.center[1] : Math.sin(i * 7));
      const predictor = mode === "hydrostatic"
        ? Float64Array.from(fixture.faces, f => f.axis === 1 ? -9.81 : 0)
        : Float64Array.from(fixture.faces, (_, k) => Math.sin(k * 13));
      const rhs = geometricDivergence(fixture, predictor).map((v, i) => -v * fixture.cells[i]!.volume);
      let pressure: Float64Array = new Float64Array(fixture.cells.length);
      for (let round = 0; round < 16; round++) pressure = deferredPressureStep(fixture, pressure, rhs, solve);
      const corrected = faceGradient(fixture, pressure).map((v, k) => predictor[k]! - v);
      const residual = geometricDivergence(fixture, corrected);
      const label = `fixture ${index}, ${mode}`;
      assert.ok(max(residual) < 1e-7, `${label}: residual ${max(residual)}`);
      if (mode === "hydrostatic") {
        assert.ok(max(corrected) < 1e-7, `${label}: parasitic velocity ${max(corrected)}`);
        const gauge = target.at(-1)!;
        assert.ok(max(pressure.map((v, i) => v - target[i]! + gauge)) < 1e-7, `${label}: pressure mismatch`);
      }
      const totalFlux = residual.reduce((sum, v, i) => sum + v * fixture.cells[i]!.volume, 0);
      assert.ok(Math.abs(totalFlux) < 1e-10, `${label}: physical flux conservation`);
    }
  }
});

test("reject the fixed outer-correction schedule for a domain only one coarse cell high", () => {
  const layout = mixedPressureLayouts().find(l => l.lattice.dimensions[1] === 4)!;
  const fixture = mixedPressureFixture(layout), solve = factorPressureCore(fixture);
  const predictor = Float64Array.from(fixture.faces, f => f.axis === 1 ? -9.81 : 0);
  const rhs = geometricDivergence(fixture, predictor).map((v, i) => -v * fixture.cells[i]!.volume);
  const equilibrium = Float64Array.from(fixture.cells, c => -9.81 * (c.center[1] - fixture.cells.at(-1)!.center[1]));
  const fixedPoint = deferredPressureStep(fixture, equilibrium, rhs, solve);
  assert.ok(max(fixedPoint.map((v, i) => v - equilibrium[i]!)) < 1e-10);
  let pressure: Float64Array = new Float64Array(fixture.cells.length);
  for (let round = 0; round < 16; round++) pressure = deferredPressureStep(fixture, pressure, rhs, solve);
  const residual = geometricDivergence(fixture, faceGradient(fixture, pressure).map((v, k) => predictor[k]! - v));
  // Preserve this counterexample: exact affine equilibrium does NOT make this
  // fixed outer schedule suitable for production. The 1e-7 acceptance target
  // is unchanged; this fixture misses it by more than four orders of magnitude.
  assert.ok(max(residual) > 1e-3);
});
