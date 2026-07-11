import assert from "node:assert/strict";
import test from "node:test";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { cloneScene, defaultScene } from "../lib/model";

function testScene(initialCondition: "dam-break" | "tank-fill" = "dam-break") {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = initialCondition;
  scene.numerics.fixedDt_s = 0.004;
  return scene;
}

test("E4-01 pressure projection reduces RMS divergence by at least 1e5", () => {
  const solver = new EulerianFluidSolver(testScene(), 1500);
  solver.setDeterministicVelocityField();
  const before = solver.computeDivergenceNorm();
  const pressure = solver.project(0.004);
  const after = solver.computeDivergenceNorm();
  assert.ok(before > 0.1);
  assert.ok(after < before * 1e-5, `${before} -> ${after}`);
  assert.ok(pressure.converged);
  assert.ok(pressure.relativeResidual <= 1e-8);
});

test("E4-02 closed-wall normal velocities remain exactly zero", () => {
  const solver = new EulerianFluidSolver(testScene(), 1500);
  solver.setDeterministicVelocityField();
  solver.step(0.004);
  const u = (i: number, j: number, k: number) => i + (solver.nx + 1) * (j + solver.ny * k);
  const v = (i: number, j: number, k: number) => i + solver.nx * (j + (solver.ny + 1) * k);
  const w = (i: number, j: number, k: number) => i + solver.nx * (j + solver.ny * k);
  for (let k = 0; k < solver.nz; k += 1) for (let j = 0; j < solver.ny; j += 1) {
    assert.equal(solver.u[u(0, j, k)], 0); assert.equal(solver.u[u(solver.nx, j, k)], 0);
  }
  for (let k = 0; k < solver.nz; k += 1) for (let i = 0; i < solver.nx; i += 1) {
    assert.equal(solver.v[v(i, 0, k)], 0); assert.equal(solver.v[v(i, solver.ny, k)], 0);
  }
  for (let j = 0; j < solver.ny; j += 1) for (let i = 0; i < solver.nx; i += 1) {
    assert.equal(solver.w[w(i, j, 0)], 0); assert.equal(solver.w[w(i, j, solver.nz)], 0);
  }
});

test("E4-03 marker volume is conserved and state remains finite", () => {
  const solver = new EulerianFluidSolver(testScene(), 1500);
  let diagnostics = solver.diagnostics;
  for (let step = 0; step < 80; step += 1) diagnostics = solver.step(0.004);
  assert.equal(diagnostics.markerVolumeDrift, 0);
  assert.equal(diagnostics.nanCount, 0);
  assert.equal(diagnostics.boundaryPenetrationCount, 0);
});

test("E4-04 dam front advances under gravity", () => {
  const solver = new EulerianFluidSolver(testScene(), 1500);
  const initialFront = solver.diagnostics.damFront_m;
  let diagnostics = solver.diagnostics;
  for (let step = 0; step < 80; step += 1) diagnostics = solver.step(0.004);
  assert.ok(diagnostics.damFront_m > initialFront + 0.12, `${initialFront} -> ${diagnostics.damFront_m}`);
  assert.ok(diagnostics.kineticEnergy_J > 1);
});

test("E4-05 deterministic replay is byte-identical", () => {
  const a = new EulerianFluidSolver(testScene(), 1200);
  const b = new EulerianFluidSolver(testScene(), 1200);
  for (let step = 0; step < 20; step += 1) { a.step(0.004); b.step(0.004); }
  assert.deepEqual(a.u, b.u);
  assert.deepEqual(a.v, b.v);
  assert.deepEqual(a.w, b.w);
  assert.deepEqual(a.pressure, b.pressure);
  assert.deepEqual(a.markers, b.markers);
  assert.deepEqual(a.fluid, b.fluid);
});

test("E4-06 adaptive time step identifies the active bound", () => {
  const solver = new EulerianFluidSolver(testScene(), 1000);
  const quiet = solver.step();
  assert.equal(quiet.limitingCondition, "user-max");
  solver.setDeterministicVelocityField();
  const moving = solver.step();
  assert.ok(moving.dt_s <= moving.advectiveLimit_s + Number.EPSILON);
  assert.ok(moving.dt_s <= moving.viscousLimit_s + Number.EPSILON);
  assert.ok(moving.dt_s <= defaultScene.numerics.maxDt_s + Number.EPSILON);
});

test("E4-07 static tank fill remains hydrostatic and energy bounded", () => {
  const scene = testScene("tank-fill");
  scene.container.fillFraction = 0.7;
  const solver = new EulerianFluidSolver(scene, 2500);
  let maxKineticEnergy = 0;
  let diagnostics = solver.diagnostics;
  for (let step = 0; step < 40; step += 1) {
    diagnostics = solver.step(0.004);
    maxKineticEnergy = Math.max(maxKineticEnergy, diagnostics.kineticEnergy_J);
  }
  assert.ok(maxKineticEnergy < 1e-8, String(maxKineticEnergy));
  assert.equal(diagnostics.occupiedVolumeDrift, 0);
  assert.equal(diagnostics.nanCount, 0);
});

test("E4-08 hydrostatic pressure converges to rho g depth", () => {
  const scene = testScene("tank-fill");
  scene.container.fillFraction = 0.7;
  const solver = new EulerianFluidSolver(scene, 5000);
  solver.step(0.004);
  let topJ = -1;
  for (let j = 0; j < solver.ny; j += 1) if (solver.fluid[solver.nx * j]) topJ = j;
  const surfaceY = (topJ + 1) * solver.hy;
  let error2 = 0, reference2 = 0;
  for (let k = 0; k < solver.nz; k += 1) for (let j = 0; j < topJ; j += 1) for (let i = 0; i < solver.nx; i += 1) {
    const q = i + solver.nx * (j + solver.ny * k);
    if (!solver.fluid[q]) continue;
    const expected = scene.fluid.density_kg_m3 * Math.abs(scene.fluid.gravity_m_s2.y) * (surfaceY - (j + 0.5) * solver.hy);
    error2 += (solver.pressure[q] - expected) ** 2;
    reference2 += expected ** 2;
  }
  const relativeL2 = Math.sqrt(error2 / reference2);
  assert.ok(relativeL2 < 0.1, String(relativeL2));
});

test("E4-09 effective grid resolution increases monotonically", () => {
  const coarse = new EulerianFluidSolver(testScene(), 600);
  const medium = new EulerianFluidSolver(testScene(), 1200);
  const fine = new EulerianFluidSolver(testScene(), 2400);
  assert.ok(coarse.nx * coarse.ny * coarse.nz < medium.nx * medium.ny * medium.nz);
  assert.ok(medium.nx * medium.ny * medium.nz < fine.nx * fine.ny * fine.nz);
  assert.ok(coarse.effectiveCellSize_m > medium.effectiveCellSize_m);
  assert.ok(medium.effectiveCellSize_m > fine.effectiveCellSize_m);
});
