import assert from "node:assert/strict";
import test from "node:test";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { inflowStrength } from "../lib/initial-fluid";
import { validateScene } from "../lib/model";
import { createPaperScenario, paperScenarios } from "../lib/paper-scenarios";
import { advanceRigidBodies, initializeRigidBodies } from "../lib/rigid-body";
import { createTallCellLayout } from "../lib/tall-cell-grid";

test("paper-derived scenarios are valid, deterministic, and uniquely identified", () => {
  assert.deepEqual(paperScenarios.map((scenario) => scenario.paperFigure), ["Figure 3", "Figure 4", "Figure 6"]);
  for (const metadata of paperScenarios) {
    const first = createPaperScenario(metadata.id), second = createPaperScenario(metadata.id);
    assert.deepEqual(first, second);
    assert.deepEqual(validateScene(first), []);
    assert.equal(new Set(first.rigidBodies.map((body) => body.id)).size, first.rigidBodies.length);
  }
});

test("inflow ramp is bounded and zero outside its configured interval", () => {
  assert.equal(inflowStrength(-1, 0, 2, 0.5), 0);
  assert.equal(inflowStrength(0, 0, 2, 0.5), 0);
  assert.equal(inflowStrength(0.25, 0, 2, 0.5), 0.5);
  assert.equal(inflowStrength(1, 0, 2, 0.5), 1);
  assert.equal(inflowStrength(1.75, 0, 2, 0.5), 0.5);
  assert.equal(inflowStrength(2, 0, 2, 0.5), 0);
});

test("hose source injects represented liquid into the CPU oracle", () => {
  const scene = createPaperScenario("hose-tank");
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.05;
  const solver = new EulerianFluidSolver(scene, 5000), initialMarkers = solver.markers.length;
  for (let step = 0; step < 8; step += 1) solver.step(0.05);
  assert.ok(solver.markers.length > initialMarkers, `${solver.markers.length} <= ${initialMarkers}`);
  assert.equal(solver.diagnostics.nanCount, 0);
});

test("hose layout retains a regular band spanning the receiving surface and nozzle", () => {
  const scene = createPaperScenario("hose-tank"), layout = createTallCellLayout(scene, "balanced");
  const fillTop = Math.floor(scene.container.fillFraction * layout.fineNy);
  const nozzle = Math.floor(scene.fluid.inflow!.center_m.y / scene.container.height_m * layout.fineNy);
  assert.ok(layout.settings.regularLayers >= nozzle - fillTop, `${layout.settings.regularLayers} < ${nozzle - fillTop}`);
  assert.ok(layout.columnBases.every((base) => base === 0));
  assert.equal(layout.settings.regularLayers, layout.fineNy, "ordinary-cell limit must retain every cubic row");
});

test("paper sphere obstacle remains fixed while dynamic bodies advance", () => {
  const scene = createPaperScenario("sphere-jet"), bodies = initializeRigidBodies(scene.rigidBodies), before = { ...bodies[0].position_m };
  advanceRigidBodies(bodies, scene, 1 / 30);
  assert.deepEqual(bodies[0].position_m, before);
  assert.deepEqual(bodies[0].linearVelocity_m_s, { x: 0, y: 0, z: 0 });
  assert.equal(bodies[0].inverseMass_kg, 0);
});
