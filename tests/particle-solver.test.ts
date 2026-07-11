import assert from "node:assert/strict";
import test from "node:test";
import { ParticleFluidSolver } from "../lib/particle-solver";
import { cloneScene, defaultScene } from "../lib/model";

function particleScene() {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "tank-fill";
  scene.container.fillFraction = 0.7;
  scene.numerics.particleSpacing_m = 0.05;
  return scene;
}

test("P5-01 uniform-grid neighbours exactly match brute force", () => {
  const solver = new ParticleFluidSolver(particleScene(), 450);
  for (let i = 0; i < solver.count; i += Math.max(1, Math.floor(solver.count / 31))) {
    assert.deepEqual(solver.optimizedNeighbours(i), solver.bruteForceNeighbours(i));
  }
});

test("P5-02 interior poly6 lattice density is within two percent", () => {
  const solver = new ParticleFluidSolver(particleScene(), 1800);
  assert.ok(solver.diagnostics.interiorMeanDensityError < 0.02, String(solver.diagnostics.interiorMeanDensityError));
});

test("P5-03 PBF positions remain within every closed boundary", () => {
  const scene = particleScene();
  const solver = new ParticleFluidSolver(scene, 500);
  for (let step = 0; step < 12; step += 1) solver.step(0.004);
  for (let i = 0; i < solver.count; i += 1) {
    const x = solver.positions[3 * i], y = solver.positions[3 * i + 1], z = solver.positions[3 * i + 2];
    assert.ok(x >= -scene.container.width_m / 2 && x <= scene.container.width_m / 2);
    assert.ok(y >= 0 && y <= scene.container.height_m);
    assert.ok(z >= -scene.container.depth_m / 2 && z <= scene.container.depth_m / 2);
  }
});

test("P5-04 particle mass and volume are exactly conserved", () => {
  const solver = new ParticleFluidSolver(particleScene(), 500);
  const count = solver.count, initialVolume = solver.initialVolume_m3;
  let diagnostics = solver.diagnostics;
  for (let step = 0; step < 12; step += 1) diagnostics = solver.step(0.004);
  assert.equal(solver.count, count);
  assert.equal(diagnostics.estimatedVolume_m3, initialVolume);
  assert.equal(diagnostics.volumeDrift, 0);
  assert.equal(diagnostics.nanCount, 0);
});

test("P5-05 same-build replay is byte-identical", () => {
  const a = new ParticleFluidSolver(particleScene(), 400);
  const b = new ParticleFluidSolver(particleScene(), 400);
  for (let step = 0; step < 8; step += 1) { a.step(0.004); b.step(0.004); }
  assert.deepEqual(a.positions, b.positions);
  assert.deepEqual(a.velocities, b.velocities);
  assert.deepEqual(a.densities, b.densities);
  assert.deepEqual(a.diagnostics, b.diagnostics);
});

test("P5-06 refinement decreases effective spacing and increases particles", () => {
  const coarse = new ParticleFluidSolver(particleScene(), 250);
  const medium = new ParticleFluidSolver(particleScene(), 500);
  const fine = new ParticleFluidSolver(particleScene(), 1000);
  assert.ok(coarse.spacing_m > medium.spacing_m && medium.spacing_m > fine.spacing_m);
  assert.ok(coarse.count < medium.count && medium.count < fine.count);
});

test("P5-07 static tank fill remains finite with bounded kinetic energy", () => {
  const scene = particleScene(), solver = new ParticleFluidSolver(scene, 400);
  let initialPotential = 0, maxKinetic = 0, diagnostics = solver.diagnostics;
  for (let i = 0; i < solver.count; i += 1) initialPotential += solver.particleMass_kg * Math.abs(scene.fluid.gravity_m_s2.y) * solver.positions[3 * i + 1];
  for (let step = 0; step < 100; step += 1) { diagnostics = solver.step(0.004); maxKinetic = Math.max(maxKinetic, diagnostics.kineticEnergy_J); }
  assert.ok(maxKinetic < 0.25 * initialPotential, `${maxKinetic} / ${initialPotential}`);
  assert.ok(diagnostics.interiorMeanDensityError < 0.02, String(diagnostics.interiorMeanDensityError));
  assert.equal(diagnostics.nanCount, 0);
});
